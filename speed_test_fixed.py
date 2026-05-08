"""Fixed version: proper resource cleanup, progress monitoring, and single-model reuse."""
import sys, time, gc, re, os, json
sys.path.insert(0, r"C:\Users\neera\OneDrive\Desktop\sep")
import qwen_dsa_pattern_spec_from_scratch as qwen_mod
import llama_cpp

MODEL_PATH = r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
CORPUS_PATH = r"C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json"
TRACE_PATH = r"C:\Users\neera\OneDrive\Desktop\structspec\Structspec\speed_trace.jsonl"
REPORT_PATH = r"C:\Users\neera\OneDrive\Desktop\structspec\Structspec\speed_report.md"

def create_miner(min_support, min_conf, det_conf):
    """Create and fit a fresh miner with given thresholds."""
    corpus = qwen_mod.QwenTokenCorpus(CORPUS_PATH)
    miner = qwen_mod.PatternMiner(
        corpus.token_text,
        max_ctx=8,
        min_support=min_support,
        min_conf=min_conf,
        det_conf=det_conf,
        min_rule_ctx=4,
    )
    miner.fit(corpus.sequences)
    return miner, corpus

def run_benchmark(miner, syntax, prompts, label="", opt_components=None):
    """Run 20 prompts and return metrics + per-prompt times."""
    ok = pg = ps = tg = ts = 0
    all_rec = []
    prompt_times = []
    opt_components = opt_components or {}
    speed_trace = opt_components.get("speed_trace")
    
    for idx, prompt in enumerate(prompts, 1):
        # Greedy baseline
        greedy_ids, greedy_text, greedy_metrics = qwen_mod.run_greedy(syntax.model, prompt, 100)

        # Start trace
        if speed_trace is not None:
            speed_trace.start(prompt_id=idx, prompt_text=prompt,
                              greedy_time=greedy_metrics["time"],
                              greedy_passes=greedy_metrics["passes"])

        # Speculative
        spec_ids, spec_text, spec_metrics = qwen_mod.run_speculative(
            syntax.model, miner, syntax, prompt, 100,
            k=6,
            reject_mode="truncate",
            **opt_components
        )

        # Accumulate
        ok += int(greedy_ids == spec_ids)
        pg += greedy_metrics["passes"]
        ps += spec_metrics["passes"]
        tg += greedy_metrics["time"]
        ts += spec_metrics["time"]
        all_rec.extend(spec_metrics.get("records", []))
        prompt_times.append((idx, greedy_metrics["time"], spec_metrics["time"],
                             greedy_metrics["passes"], spec_metrics["passes"],
                             greedy_ids == spec_ids))

        if idx % 5 == 0:
            speed = pg / max(1, ps)
            print(f"  [{idx:02d}/20] passes: {pg}/{ps} = {speed:.3f}×  matches: {ok}/{idx}")

    # Derived metrics
    cov = sum(1 for r in all_rec if r["draft"] > 0) / len(all_rec) if all_rec else 0.0
    prop_tokens = sum(int(r["draft"]) for r in all_rec)
    acc_tokens = sum(int(r["accepted_draft"]) for r in all_rec)
    acc = acc_tokens / max(1, prop_tokens)

    return {
        "pass_speedup": pg / max(1, ps),
        "time_speedup": tg / max(1e-9, ts),
        "match_rate": ok / 20,
        "coverage": cov,
        "accuracy": acc,
        "rules": len(miner.rules_by_ctx),
        "prompt_times": prompt_times,
        "ok": ok,
        "total_passes_g": pg,
        "total_passes_s": ps,
    }

def main():
    print("="*80)
    print("STRUCTSPEC SPEED TEST — clean, fixed, no leaks + OPTIMIZATIONS")
    print("Model: Qwen2.5-7B-Instruct Q4_K_M")
    print("Corpus: engineering_dsa_tokens (20 examples)")
    print("Optimizations: METHOD1-9 integrated")
    print("="*80)
    
    prompts = qwen_mod.PROMPTS[:20]
    
    # Clean trace file
    import os
    if os.path.exists(TRACE_PATH):
        os.remove(TRACE_PATH)
    
    # ── Config 1: Strict thresholds ─────────────────────────────────────
    print("\n[1/2] STRICT thresholds (min_support=2, min_conf=0.96, det_conf=0.96)")
    miner1, corpus1 = create_miner(2, 0.96, 0.96)
    print(f"  Mined {len(miner1.rules_by_ctx)} rules")
    model1 = qwen_mod.FastGreedyLlama(MODEL_PATH, n_ctx=2048, n_gpu_layers=-1)
    syntax1 = qwen_mod.PythonSyntaxProposer(model1, mode="cluster")
    
    # Optimization components (shared across prompts)
    opt1 = {
        "rule_registry": qwen_mod.RuleStatsRegistry(),
        "adaptive_k": qwen_mod.AdaptiveKController(k_init=6, k_min=1, k_max=12),
        "top_k_filter": qwen_mod.LastTargetTopKFilter(top_k=5),
        "early_exit": qwen_mod.EarlyExitDraftController(),
        "entropy_abort": qwen_mod.EntropyProxyAbort(),
        "speed_trace": qwen_mod.SpeedTraceCollector(TRACE_PATH),
    }
    
    t0 = time.perf_counter()
    r1 = run_benchmark(miner1, syntax1, prompts, label="strict", opt_components=opt1)
    t1 = time.perf_counter()
    print(f"  Runtime: {t1-t0:.2f}s")
    opt1["speed_trace"].close()
    
    # Store rule count before cleanup
    strict_rules = len(miner1.rules_by_ctx)
    # Cleanup
    del miner1, corpus1, model1, syntax1, opt1
    gc.collect()
    
    # ── Config 2: Loose thresholds ──────────────────────────────────────
    print("\n[2/2] LOOSE thresholds (min_support=1, min_conf=0.88, det_conf=0.92)")
    miner2, corpus2 = create_miner(1, 0.88, 0.92)
    print(f"  Mined {len(miner2.rules_by_ctx)} rules")
    model2 = qwen_mod.FastGreedyLlama(MODEL_PATH, n_ctx=2048, n_gpu_layers=-1)
    syntax2 = qwen_mod.PythonSyntaxProposer(model2, mode="cluster")
    
    opt2 = {
        "rule_registry": qwen_mod.RuleStatsRegistry(),
        "adaptive_k": qwen_mod.AdaptiveKController(k_init=6, k_min=1, k_max=12),
        "top_k_filter": qwen_mod.LastTargetTopKFilter(top_k=5),
        "early_exit": qwen_mod.EarlyExitDraftController(),
        "entropy_abort": qwen_mod.EntropyProxyAbort(),
        "speed_trace": qwen_mod.SpeedTraceCollector(TRACE_PATH),
    }
    
    t2 = time.perf_counter()
    r2 = run_benchmark(miner2, syntax2, prompts, label="loose", opt_components=opt2)
    t3 = time.perf_counter()
    print(f"  Runtime: {t3-t2:.2f}s")
    opt2["speed_trace"].close()
    
    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Config':<25} {'Pass↑':>9} {'Time↑':>9} {'Match':>7} {'Cover':>7} {'Acc':>7}")
    print("-"*80)
    print(f"{'Strict (2,0.96,0.96)':<25} {r1['pass_speedup']:>8.3f}× {r1['time_speedup']:>8.3f}× {r1['match_rate']*100:>5.1f}% {r1['coverage']*100:>5.1f}% {r1['accuracy']*100:>5.1f}%")
    print(f"{'Loose (1,0.88,0.92)':<25} {r2['pass_speedup']:>8.3f}× {r2['time_speedup']:>8.3f}× {r2['match_rate']*100:>5.1f}% {r2['coverage']*100:>5.1f}% {r2['accuracy']*100:>5.1f}%")
    print()
    print(f"IMPROVEMENT:  Pass +{r2['pass_speedup'] - r1['pass_speedup']:.3f}×   Time +{r2['time_speedup'] - r1['time_speedup']:.3f}×")
    print(f"  Rules increase: {len(miner2.rules_by_ctx)} vs {strict_rules} (+{len(miner2.rules_by_ctx)-strict_rules})")
    print(f"  Coverage increase: {r2['coverage']*100:.1f}% vs {r1['coverage']*100:.1f}% (+{(r2['coverage']-r1['coverage'])*100:.1f}pp)")

if __name__ == "__main__":
    main()
