"""
spec_decode_rejection_optimizer.py
====================================
All 9 rejection-time reduction methods from the research doc,
implemented as drop-in components for a rule-based spec decode system.

Methods implemented:
  1. RecoveryModeSelector     — truncate vs seq-bonus switcher
  2. EarlyExitDraftController — stop drafting when confidence drops
  3. RuleStatsTracker         — per-rule cooldown / pre-rejection
  4. TokenContextManager      — O(1) context rebuild, no re-encode
  5. GreedyCorrectionSampler  — skip residual distribution (greedy only)
  6. LastTargetTopKFilter     — pre-verify guidance from cached logits
  7. EntropyProxyAbort        — rule-count entropy proxy, adaptive k
  8. AdaptiveKController      — rolling acceptance rate → dynamic k
  9. BenchmarkLogger          — buffered logs, no hot-path prints
"""

from __future__ import annotations
import time
import collections
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch


# ─────────────────────────────────────────────
# METHOD 1 — Truncate vs Seq-Bonus Selector
# ─────────────────────────────────────────────
# Research finding (PEARL + SpecBranch papers):
#   truncate beats seq-bonus in wall time when accept rate < ~90%
#
# Seq-bonus: on rejection at i, runs an EXTRA decode pass to get token i+1
#   → costs +1 decode step per rejection
# Truncate: takes correction token at i, restarts from i+1
#   → no extra decode
#
# Since your acceptance rate is 86.5% (reject rate ~13.5%),
# truncate is almost certainly faster. This class lets you A/B test both
# and automatically picks the winner after a calibration window.

class RecoveryModeSelector:
    """
    Measures wall time per accepted token for truncate vs seq-bonus.
    After `calibration_rounds` rounds, locks to the faster mode.

    Usage:
        selector = RecoveryModeSelector()
        mode = selector.get_mode()          # "truncate" or "seq_bonus"
        t0 = time.perf_counter()
        # ... run your recovery logic using mode ...
        selector.record(mode, accepted_tokens, time.perf_counter() - t0)
    """
    MODES = ("truncate", "seq_bonus")

    def __init__(self, calibration_rounds: int = 40):
        self.calibration_rounds = calibration_rounds
        self._stats: Dict[str, Dict] = {
            m: {"total_time": 0.0, "total_tokens": 0, "rounds": 0}
            for m in self.MODES
        }
        self._round = 0
        self._locked_mode: Optional[str] = None

    def get_mode(self) -> str:
        if self._locked_mode:
            return self._locked_mode
        # Alternate between modes during calibration
        return self.MODES[self._round % 2]

    def record(self, mode: str, accepted_tokens: int, elapsed: float):
        if self._locked_mode:
            return
        s = self._stats[mode]
        s["total_time"] += elapsed
        s["total_tokens"] += max(accepted_tokens, 1)
        s["rounds"] += 1
        self._round += 1

        if self._round >= self.calibration_rounds:
            self._lock()

    def _lock(self):
        # Lower ms/token → faster mode
        scores = {}
        for m, s in self._stats.items():
            if s["total_tokens"] > 0:
                scores[m] = s["total_time"] / s["total_tokens"]
        if scores:
            self._locked_mode = min(scores, key=scores.get)
            print(f"[RecoveryMode] Locked to: {self._locked_mode} "
                  f"| scores: { {k: f'{v*1000:.2f}ms/tok' for k,v in scores.items()} }")

    def recovery_truncate(
        self,
        context_token_ids: List[int],
        accepted_end: int,
        correction_token_id: int,
    ) -> List[int]:
        """
        Truncate mode: slice context to accepted prefix + correction token.
        Pure list slice — O(1) effectively, no tokenizer call.
        """
        new_context = context_token_ids[:accepted_end]
        new_context.append(correction_token_id)
        return new_context

    def recovery_seq_bonus(
        self,
        context_token_ids: List[int],
        accepted_end: int,
        correction_token_id: int,
        run_extra_decode_fn,          # callable → int (token_id)
    ) -> List[int]:
        """
        Seq-bonus mode: truncate + run one extra decode to get i+1.
        Only worth it if acceptance rate > ~90%.
        run_extra_decode_fn: your target model's single-step decode,
                             takes context token ids, returns next token id.
        """
        new_context = context_token_ids[:accepted_end]
        new_context.append(correction_token_id)
        bonus_token = run_extra_decode_fn(new_context)
        new_context.append(bonus_token)
        return new_context


# ─────────────────────────────────────────────
# METHOD 2 — Early-Exit Draft Controller
# ─────────────────────────────────────────────
# Research: EESD paper (2024). Fixed k is suboptimal.
# "Drafting more tokens when confidence drops just increases future rejections."
# Modelled as a Bernoulli process: should I draft one more token?

class EarlyExitDraftController:
    """
    Before each draft token extension, decide whether to keep going.
    Returns False → stop, submit what you have to verify now.

    Usage:
        controller = EarlyExitDraftController()
        for pos in range(k_max):
            if not controller.should_extend(rule, pos, rule_stats):
                break
            draft_tokens.append(rule.next_token(pos))
    """

    def __init__(
        self,
        min_rule_accept_rate: float = 0.80,
        min_rule_conf: float = 0.88,
        high_risk_position_threshold: float = 0.30,
        conf_check_after_position: int = 2,
    ):
        self.min_accept = min_rule_accept_rate
        self.min_conf = min_rule_conf
        self.risk_thresh = high_risk_position_threshold
        self.conf_after = conf_check_after_position

        # Tracks rejection frequency per (rule_pattern_hash, position)
        # format: { (hash, pos): (reject_count, fire_count) }
        self._position_risk: Dict[Tuple, Tuple[int, int]] = {}

    def should_extend(
        self,
        rule_pattern_hash: int,
        rule_conf: float,
        rule_recent_accept_rate: float,
        current_position: int,
    ) -> bool:
        """True = keep drafting. False = stop now, go verify."""

        # Gate 1: rolling acceptance rate (live signal)
        if rule_recent_accept_rate < self.min_accept:
            return False

        # Gate 2: static rule confidence drops after position 2
        if current_position >= self.conf_after and rule_conf < self.min_conf:
            return False

        # Gate 3: this specific position has historically been a rejection point
        key = (rule_pattern_hash, current_position)
        if key in self._position_risk:
            rejects, fires = self._position_risk[key]
            if fires > 5 and (rejects / fires) > self.risk_thresh:
                return False

        return True

    def record_outcome(
        self,
        rule_pattern_hash: int,
        position: int,
        was_rejected: bool,
    ):
        """Call after each verify to update position risk table."""
        key = (rule_pattern_hash, position)
        rejects, fires = self._position_risk.get(key, (0, 0))
        self._position_risk[key] = (
            rejects + (1 if was_rejected else 0),
            fires + 1,
        )


# ─────────────────────────────────────────────
# METHOD 3 — Per-Rule Stats + Cooldown
# ─────────────────────────────────────────────
# Core insight: a rule that got rejected twice in a row should NOT fire again.
# Each firing of a bad rule costs exactly as much as greedy but adds Python overhead.
# So: bad rule firing = greedy cost + overhead. Bad rule is WORSE than just greedy.

@dataclass
class RuleStats:
    name: str
    offline_conf: float            # confidence from your pattern miner

    fires: int = 0
    accepts: int = 0
    rejects: int = 0
    consecutive_rejects: int = 0
    last_five: List[int] = field(default_factory=list)  # 1=accept, 0=reject
    reject_at_position: Dict[int, int] = field(default_factory=dict)

    # Cooldown state
    cooldown_remaining: int = 0

    @property
    def live_acceptance_rate(self) -> float:
        if len(self.last_five) < 3:
            return self.offline_conf   # fall back to miner confidence
        return sum(self.last_five) / len(self.last_five)

    def should_fire(self) -> bool:
        """
        Pre-rejection gate — call this BEFORE building the draft.
        Returns False → skip spec decode, do greedy instead.
        """
        if self.cooldown_remaining > 0:
            return False
        if self.consecutive_rejects >= 2:
            return False
        if self.live_acceptance_rate < 0.70:
            return False
        return True

    def record_accept(self, position: int):
        self.fires += 1
        self.accepts += 1
        self.consecutive_rejects = 0
        self.last_five.append(1)
        if len(self.last_five) > 5:
            self.last_five.pop(0)

    def record_reject(self, position: int):
        self.fires += 1
        self.rejects += 1
        self.consecutive_rejects += 1
        self.last_five.append(0)
        if len(self.last_five) > 5:
            self.last_five.pop(0)
        self.reject_at_position[position] = (
            self.reject_at_position.get(position, 0) + 1
        )
        # Auto-cooldown after 2+ consecutive rejections
        if self.consecutive_rejects >= 2:
            self.cooldown_remaining = 5   # skip next 5 opportunities

    def tick_cooldown(self):
        """Call once per decode step to count down cooldown."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1


class RuleStatsRegistry:
    """Holds RuleStats for all rules, provides lookup."""

    def __init__(self):
        self._registry: Dict[str, RuleStats] = {}

    def get_or_create(self, rule_name: str, offline_conf: float) -> RuleStats:
        if rule_name not in self._registry:
            self._registry[rule_name] = RuleStats(
                name=rule_name, offline_conf=offline_conf
            )
        return self._registry[rule_name]

    def tick_all(self):
        for stats in self._registry.values():
            stats.tick_cooldown()

    def summary(self) -> Dict[str, dict]:
        return {
            name: {
                "accept_rate": s.live_acceptance_rate,
                "consecutive_rejects": s.consecutive_rejects,
                "cooldown": s.cooldown_remaining,
                "fires": s.fires,
            }
            for name, s in self._registry.items()
        }


# ─────────────────────────────────────────────
# METHOD 4 — Token Context Manager (O(1) Rebuild)
# ─────────────────────────────────────────────
# Problem: after rejection, many systems do:
#   text = tokenizer.decode(all_ids)
#   ids  = tokenizer.encode(text)     ← O(n) + Python overhead every time
#
# Fix: maintain a token-id list directly. Rejection = one list slice + append.
# Also cache token_id → text so no repeated tokenizer calls in hot loop.

class TokenContextManager:
    """
    Maintains generation context as a flat token-id list.
    Never calls tokenizer.decode/encode inside the rejection path.

    Usage:
        ctx = TokenContextManager(tokenizer, prompt_ids)

        # After verify — accepted 3 tokens, rejected at position 3,
        # correction token is 4521:
        ctx.apply_rejection(accepted_end_pos=3, correction_token_id=4521)

        # After full accept (all k tokens accepted + bonus):
        ctx.apply_accept(accepted_token_ids=[...], bonus_token_id=999)

        # Get context for next draft:
        ids = ctx.token_ids
    """

    def __init__(self, tokenizer, prompt_ids: List[int]):
        self._tokenizer = tokenizer
        self.token_ids: List[int] = list(prompt_ids)
        self._text_cache: Dict[int, str] = {}

    def apply_rejection(self, accepted_end_pos: int, correction_token_id: int):
        """
        O(1): slice to accepted prefix, append correction token.
        No tokenizer call.
        """
        # accepted_end_pos is absolute index into self.token_ids
        self.token_ids = self.token_ids[:accepted_end_pos]
        self.token_ids.append(correction_token_id)

    def apply_accept(self, accepted_token_ids: List[int], bonus_token_id: Optional[int] = None):
        """Extend context with accepted draft tokens + optional bonus."""
        self.token_ids.extend(accepted_token_ids)
        if bonus_token_id is not None:
            self.token_ids.append(bonus_token_id)

    def token_to_text(self, token_id: int) -> str:
        """Cached single-token decode. Avoids repeat tokenizer calls."""
        if token_id not in self._text_cache:
            self._text_cache[token_id] = self._tokenizer.decode([token_id])
        return self._text_cache[token_id]

    def decode_all(self) -> str:
        """Full decode — call only at end of generation, not in hot path."""
        return self._tokenizer.decode(self.token_ids)

    @property
    def length(self) -> int:
        return len(self.token_ids)


# ─────────────────────────────────────────────
# METHOD 5 — Greedy Correction Sampler
# ─────────────────────────────────────────────
# For greedy (temperature=0), the residual distribution collapses to argmax.
# norm(max(0, p_target - p_draft)) → just target_top_token
# No softmax, no residual computation, no norm(max()) call needed.
#
# This is valid ONLY for greedy decoding. Since you benchmark with greedy,
# this applies directly.

class GreedyCorrectionSampler:
    """
    Fast correction token sampling for greedy spec decode.

    Standard spec decode (stochastic):
        residual = norm(max(0, p_target(x) - p_draft(x)))
        correction = sample(residual)

    Greedy shortcut:
        correction = argmax(target_logits)

    Usage:
        sampler = GreedyCorrectionSampler()
        correction_id = sampler.get_correction(target_logits_at_position_i, draft_token_id)
    """

    def get_correction(
        self,
        target_logits: torch.Tensor,   # shape: (vocab_size,)
        draft_token_id: int,
    ) -> int:
        """
        Returns the correction token id.
        For greedy: always returns target argmax.
        No softmax, no residual distribution, just argmax.
        """
        return int(target_logits.argmax().item())

    def batch_verify_greedy(
        self,
        target_logits: torch.Tensor,   # shape: (seq_len, vocab_size)
        draft_token_ids: List[int],
        context_length: int,           # length of context before draft
    ) -> Tuple[int, int, int]:
        """
        Verifies all draft tokens in one pass (greedy).
        Returns (num_accepted, first_reject_pos, correction_token_id)

        target_logits: logits from target model over the full extended sequence.
        draft_token_ids: the k proposed tokens.
        context_length: number of tokens in context before draft starts.
        """
        num_accepted = 0
        correction_token_id = -1

        for i, draft_tok in enumerate(draft_token_ids):
            pos = context_length + i
            target_top = int(target_logits[pos].argmax().item())
            if target_top == draft_tok:
                num_accepted += 1
            else:
                # Rejection at position i
                correction_token_id = target_top
                return num_accepted, i, correction_token_id

        # All accepted — bonus token from last position
        bonus_pos = context_length + len(draft_token_ids)
        bonus_token = int(target_logits[bonus_pos].argmax().item())
        return len(draft_token_ids), -1, bonus_token   # -1 = no rejection


# ─────────────────────────────────────────────
# METHOD 6 — Last Target Top-K Filter (Pre-Verify Guidance)
# ─────────────────────────────────────────────
# From: Double paper (arXiv 2601.05524, Jan 2026)
# "Target performs single-step retrieval to generate multi-token guidance
#  which draft uses to avoid drafting tokens that will be rejected."
#
# Free implementation: you already HAVE the last target logits from the
# previous verify pass. Cache top-5. Pre-filter next draft's first token.
#
# Cost: zero extra GPU work. Pure Python dict lookup.

class LastTargetTopKFilter:
    """
    Caches the top-K token ids from the last target verify pass.
    Before drafting, filters rules whose first token is not in top-K.

    Usage:
        filter = LastTargetTopKFilter(top_k=5)

        # After each verify pass, update the cache:
        filter.update(target_logits_at_last_position)

        # Before firing a rule, check:
        if not filter.first_token_plausible(rule.draft_tokens[0]):
            skip the rule  # target almost certainly won't accept it
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._top_ids: set = set()
        self._initialized = False

    def update(self, target_logits_last_pos: torch.Tensor):
        """
        Call after every verify pass with logits at the last accepted position.
        target_logits_last_pos: shape (vocab_size,)
        """
        top_ids = target_logits_last_pos.topk(self.top_k).indices.tolist()
        self._top_ids = set(top_ids)
        self._initialized = True

    def first_token_plausible(self, first_draft_token_id: int) -> bool:
        """
        Returns True if target's top-K includes this token → safe to draft.
        Returns False → target will almost certainly reject → skip the rule.
        If not yet initialized, allow all (conservative).
        """
        if not self._initialized:
            return True
        return first_draft_token_id in self._top_ids

    def filter_rules(self, candidate_rules: list) -> list:
        """
        Filters a list of rule objects, keeping only those whose first
        draft token is in the last target top-K.
        Assumes each rule has a `first_token_id` attribute.
        """
        if not self._initialized:
            return candidate_rules
        return [r for r in candidate_rules if self.first_token_plausible(r.first_token_id)]


# ─────────────────────────────────────────────
# METHOD 7 — Entropy Proxy Abort + Adaptive k
# ─────────────────────────────────────────────
# From: Entropy-Aware Speculative Decoding (OpenReview 2025)
# "High entropy AND substantial overlap → token is uncertain → prone to errors."
#
# For a rule-based system without model logits available pre-draft:
# proxy entropy using rule count + confidence variance.

class EntropyProxyAbort:
    """
    Estimates context predictability from competing rules and their confidence.
    Returns a draft policy: "full_k", "short_k", or "greedy".

    Usage:
        abort = EntropyProxyAbort()
        policy = abort.assess(matching_rules)
        if policy == "greedy":
            # do greedy decode, skip spec
        elif policy == "short_k":
            k = 2
        else:
            k = your_full_k
    """

    def __init__(
        self,
        max_rules_before_ambiguous: int = 5,
        min_best_conf: float = 0.85,
        no_rules_fallback: str = "greedy",
    ):
        self.max_rules = max_rules_before_ambiguous
        self.min_best_conf = min_best_conf
        self.fallback = no_rules_fallback

    def assess(self, matching_rules: list) -> str:
        """
        matching_rules: list of rule objects with a `.conf` attribute.
        Returns: "full_k" | "short_k" | "greedy"
        """
        if len(matching_rules) == 0:
            return self.fallback                      # no signal → greedy

        if len(matching_rules) > self.max_rules:
            return "short_k"                          # many competing → ambiguous

        best_conf = max(r.conf for r in matching_rules)
        if best_conf < self.min_best_conf:
            return "short_k"                          # best rule is uncertain

        return "full_k"


# ─────────────────────────────────────────────
# METHOD 8 — Adaptive K Controller
# ─────────────────────────────────────────────
# From: PEARL paper + vLLM DynamicProposer (PR #26504)
# vLLM's actual implementation: monitors per-request acceptance rate,
# increases/decreases k to keep rate near a target threshold.
# "Average 1.94× across all batch sizes vs degradation with fixed k."

class AdaptiveKController:
    """
    Dynamically adjusts draft length k based on rolling acceptance rate.
    Mirrors vLLM's DynamicProposer logic (PR #26504).

    Usage:
        ctrl = AdaptiveKController(k_init=4, k_min=1, k_max=10)

        k = ctrl.get_k()
        # ... draft k tokens, run verify ...
        ctrl.update(accepted_count=3, proposed_count=k)

        # If draft was rejected at position 0 (full rejection):
        ctrl.full_rejection_penalty()
    """

    def __init__(
        self,
        k_init: int = 4,
        k_min: int = 1,
        k_max: int = 10,
        window_size: int = 10,
        up_threshold: float = 0.93,    # increase k if accept rate > this
        down_threshold: float = 0.80,  # decrease k if accept rate < this
    ):
        self.k = k_init
        self.k_min = k_min
        self.k_max = k_max
        self.window_size = window_size
        self.up_thresh = up_threshold
        self.down_thresh = down_threshold
        self._window: collections.deque = collections.deque(maxlen=window_size)

    def get_k(self) -> int:
        return self.k

    def update(self, accepted_count: int, proposed_count: int):
        """Call after each verify pass."""
        if proposed_count == 0:
            return
        rate = accepted_count / proposed_count
        self._window.append(rate)

        if len(self._window) < 3:
            return   # not enough data yet

        avg = sum(self._window) / len(self._window)

        if avg > self.up_thresh:
            self.k = min(self.k + 1, self.k_max)
        elif avg < self.down_thresh:
            self.k = max(self.k - 1, self.k_min)
        # else: k stays same

    def full_rejection_penalty(self):
        """Draft rejected at token 0 → immediately drop k by 2."""
        self.k = max(self.k_min, self.k - 2)

    @property
    def current_avg_acceptance(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)


# ─────────────────────────────────────────────
# METHOD 9 — Benchmark Logger (No Hot-Path Prints)
# ─────────────────────────────────────────────
# From: NVIDIA + PyTorch profiling docs.
# Every print(), rich render, or tokenizer.decode for display in the hot loop
# adds 0.01–0.05s per step. At 1452 spec passes, this is significant.

class BenchmarkLogger:
    """
    Buffers all log messages during generation.
    Flushes only after full generation completes.
    Never calls tokenizer.decode inside the generation loop.

    Usage:
        logger = BenchmarkLogger(enabled=True)

        # During generation (fast path):
        logger.log(f"step {i}: accepted {n}")

        # After generation completes:
        logger.flush()
        final_text = tokenizer.decode(all_generated_ids)
        print(final_text)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._buffer: List[str] = []
        self._timings: List[float] = []
        self._t_start: Optional[float] = None

    def start(self):
        self._t_start = time.perf_counter()

    def log(self, msg: str):
        if self.enabled:
            self._buffer.append(msg)

    def record_step_time(self, elapsed: float):
        self._timings.append(elapsed)

    def flush(self):
        """Call once after generation loop exits."""
        if not self.enabled:
            return
        for msg in self._buffer:
            print(msg)
        self._buffer.clear()
        if self._timings:
            avg = sum(self._timings) / len(self._timings)
            total = sum(self._timings)
            print(f"\n[BenchLog] Steps: {len(self._timings)} | "
                  f"Avg: {avg*1000:.1f}ms | Total: {total:.3f}s")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.flush()


# ─────────────────────────────────────────────
# INTEGRATION EXAMPLE — Full Decode Loop
# ─────────────────────────────────────────────
# Shows how all 9 components wire together in a single generation loop.
# Replace `your_draft_rules`, `target_model`, `tokenizer` with your own.

def optimized_spec_decode_loop(
    prompt_ids: List[int],
    target_model,
    tokenizer,
    draft_rules,           # list of rule objects
    rule_registry: RuleStatsRegistry,
    max_new_tokens: int = 200,
) -> str:

    # --- Init all components ---
    ctx         = TokenContextManager(tokenizer, prompt_ids)
    recovery    = RecoveryModeSelector()
    early_exit  = EarlyExitDraftController()
    top_k_filt  = LastTargetTopKFilter(top_k=5)
    entropy_ab  = EntropyProxyAbort()
    adaptive_k  = AdaptiveKController(k_init=4, k_min=1, k_max=8)
    sampler     = GreedyCorrectionSampler()
    logger      = BenchmarkLogger(enabled=True)

    generated_ids: List[int] = []
    step = 0

    with logger:
        while len(generated_ids) < max_new_tokens:
            t0 = time.perf_counter()

            # ── Step 1: Pre-filter rules by last target top-K ──
            candidate_rules = top_k_filt.filter_rules(draft_rules)

            # ── Step 2: Entropy proxy — check if context is predictable ──
            policy = entropy_ab.assess(candidate_rules)
            if policy == "greedy":
                # Skip spec, do greedy
                logits = target_model(ctx.token_ids)
                next_id = int(logits[-1].argmax().item())
                ctx.apply_accept([next_id])
                generated_ids.append(next_id)
                rule_registry.tick_all()
                logger.log(f"[step {step}] greedy (entropy abort)")
                logger.record_step_time(time.perf_counter() - t0)
                step += 1
                continue

            k_now = adaptive_k.get_k()
            if policy == "short_k":
                k_now = min(k_now, 2)

            # ── Step 3: Per-rule cooldown check + early-exit drafting ──
            draft_tokens: List[int] = []
            fired_rule = None

            for rule in candidate_rules:
                stats = rule_registry.get_or_create(rule.name, rule.conf)
                if not stats.should_fire():
                    continue

                # Build draft with early-exit
                rule_draft = []
                for pos in range(k_now):
                    if not early_exit.should_extend(
                        rule_pattern_hash=hash(rule.pattern),
                        rule_conf=rule.conf,
                        rule_recent_accept_rate=stats.live_acceptance_rate,
                        current_position=pos,
                    ):
                        break
                    tok = rule.get_token(pos)
                    if tok is None:
                        break
                    rule_draft.append(tok)

                if rule_draft:
                    draft_tokens = rule_draft
                    fired_rule = rule
                    break

            if not draft_tokens or fired_rule is None:
                # No rule fired — greedy fallback
                logits = target_model(ctx.token_ids)
                next_id = int(logits[-1].argmax().item())
                ctx.apply_accept([next_id])
                generated_ids.append(next_id)
                rule_registry.tick_all()
                logger.record_step_time(time.perf_counter() - t0)
                step += 1
                continue

            # ── Step 4: Run target verify pass ──
            extended_ids = ctx.token_ids + draft_tokens
            target_logits = target_model(extended_ids)  # shape: (len, vocab)

            # Update top-K cache for next draft's pre-filter
            top_k_filt.update(target_logits[len(ctx.token_ids) - 1])

            # ── Step 5: Greedy batch verify (no residual distribution) ──
            num_accepted, reject_pos, correction_or_bonus = sampler.batch_verify_greedy(
                target_logits, draft_tokens, len(ctx.token_ids)
            )

            stats = rule_registry.get_or_create(fired_rule.name, fired_rule.conf)
            rule_registry.tick_all()

            if reject_pos == -1:
                # ── ALL ACCEPTED + bonus token ──
                ctx.apply_accept(draft_tokens, bonus_token_id=correction_or_bonus)
                generated_ids.extend(draft_tokens)
                generated_ids.append(correction_or_bonus)
                adaptive_k.update(len(draft_tokens), len(draft_tokens))
                for pos in range(len(draft_tokens)):
                    stats.record_accept(pos)
                    early_exit.record_outcome(hash(fired_rule.pattern), pos, False)
                logger.log(f"[step {step}] FULL ACCEPT k={len(draft_tokens)} + bonus")

            else:
                # ── REJECTION at reject_pos ──
                accepted_end = len(ctx.token_ids) + reject_pos
                mode = recovery.get_mode()

                t_recovery = time.perf_counter()
                if mode == "truncate":
                    ctx.token_ids = recovery.recovery_truncate(
                        ctx.token_ids, accepted_end, correction_or_bonus
                    )
                else:
                    ctx.token_ids = recovery.recovery_seq_bonus(
                        ctx.token_ids, accepted_end, correction_or_bonus,
                        run_extra_decode_fn=lambda ids: int(
                            target_model(ids)[-1].argmax().item()
                        ),
                    )
                elapsed_recovery = time.perf_counter() - t_recovery

                if reject_pos == 0:
                    adaptive_k.full_rejection_penalty()
                    stats.record_reject(reject_pos)
                    early_exit.record_outcome(hash(fired_rule.pattern), reject_pos, True)
                else:
                    adaptive_k.update(reject_pos, len(draft_tokens))
                    for pos in range(reject_pos):
                        stats.record_accept(pos)
                        early_exit.record_outcome(hash(fired_rule.pattern), pos, False)
                    stats.record_reject(reject_pos)
                    early_exit.record_outcome(hash(fired_rule.pattern), reject_pos, True)

                generated_ids.append(correction_or_bonus)

                recovery.record(mode, reject_pos + 1, elapsed_recovery)
                logger.log(f"[step {step}] REJECT at pos={reject_pos} "
                           f"mode={mode} k_now={adaptive_k.get_k()}")

            logger.record_step_time(time.perf_counter() - t0)
            step += 1

    # ── Decode ONCE at the end, not in the loop ──
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────
# BENCHMARK HARNESS — measure each method's impact independently
# ─────────────────────────────────────────────

class RejectionTimeBenchmark:
    """
    A/B test harness. Run with/without each optimization enabled.
    Measures: tokens/sec, rejection rate, avg accepted per verify pass.

    Usage:
        bench = RejectionTimeBenchmark()
        with bench.measure("truncate_only"):
            # run your generation loop here
        bench.report()
    """

    def __init__(self):
        self._results: Dict[str, dict] = {}
        self._current_label: Optional[str] = None
        self._t0: float = 0

    class _Measure:
        def __init__(self, bench, label):
            self._bench = bench
            self._label = label
        def __enter__(self):
            self._bench._current_label = self._label
            self._bench._t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self._bench._t0
            if self._label not in self._bench._results:
                self._bench._results[self._label] = {}
            self._bench._results[self._label]["wall_time"] = elapsed

    def measure(self, label: str):
        return self._Measure(self, label)

    def record(self, label: str, **kwargs):
        if label not in self._results:
            self._results[label] = {}
        self._results[label].update(kwargs)

    def report(self):
        print("\n" + "="*60)
        print("REJECTION OPTIMIZER BENCHMARK REPORT")
        print("="*60)
        for label, data in self._results.items():
            print(f"\n[{label}]")
            for k, v in data.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        print("="*60)
