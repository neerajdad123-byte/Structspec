"""Accurate phase timing — properly summing ALL decode phases."""
import sys, time, json, csv, re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from llama_cpp import Llama, LlamaCache

sys.path.insert(0, r"C:\Users\neera\OneDrive\Desktop\sep")
import qwen_dsa_pattern_spec_from_scratch as qwen_mod

qwen_mod.DEFAULT_MODEL = r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
qwen_mod.DEFAULT_JSON = r"C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json"

def run_greedy_full_timing(model, prompt, max_tokens):
    model.reset()
    prompt_ids = model.tokenize(prompt)
    t0 = time.perf_counter()
    logits = model.decode_logits(prompt_ids, logits_all=False)[0]
    prompt_decode_time = time.perf_counter() - t0
    prev_pred = model.argmax(logits)
    
    gen = list(prompt_ids)
    passes = 1
    target = len(prompt_ids) + max_tokens
    decode_times = []
    start = time.perf_counter()
    
    while len(gen) < target:
        tok = prev_pred
        gen.append(tok)
        if tok == model.eos or len(gen) >= target:
            break
        t0 = time.perf_counter()
        logits = model.decode_logits([tok], logits_all=False)[0]
        decode_times.append(time.perf_counter() - t0)
        passes += 1
        prev_pred = model.argmax(logits)
    
    elapsed = time.perf_counter() - start
    out_ids = gen[:target]
    t0 = time.perf_counter()
    _ = model.detokenize(out_ids[len(prompt_ids):])
    detok_time = time.perf_counter() - t0
    
    return {
        "passes": passes,
        "time": elapsed,
        "times": {
            "prompt_decode": prompt_decode_time,
            "per_token_decode": decode_times,
            "total_decode": sum(decode_times),
            "detok": detok_time,
        }
    }

def run_spec_full_timing(model, miner, syntax, prompt, max_tokens, k=6, reject_mode="seq-bonus"):
    model.reset()
    prompt_ids = model.tokenize(prompt)
    t0 = time.perf_counter()
    logits = model.decode_logits(prompt_ids, logits_all=False)[0]
    prompt_decode_time = time.perf_counter() - t0
    prev_pred = model.argmax(logits)
    
    gen = list(prompt_ids)
    kv_len = len(prompt_ids)
    target = len(prompt_ids) + max_tokens
    passes = 1
    
    pattern_times = []
    decode_batch_times = []
    verify_times = []
    reject_times = []
    
    start = time.perf_counter()
    while len(gen) < target:
        if gen and gen[-1] == model.eos:
            break
        remaining = target - len(gen)
        if remaining <= 0:
            break
        pending = gen[kv_len:]
        max_draft = max(0, min(k, remaining - 1))
        
        # PATTERN
        t0p = time.perf_counter()
        draft, rules = qwen_mod.propose_draft(miner, syntax, gen, max_draft, banned=None)
        pattern_times.append(time.perf_counter() - t0p)
        
        if not pending and not draft:
            gen.append(prev_pred)
            if prev_pred == model.eos or len(gen) >= target:
                break
            continue
        
        batch = pending + draft
        if not batch:
            gen.append(prev_pred)
            continue
        
        old_kv = kv_len
        
        # DECODE BATCH
        t0d = time.perf_counter()
        batch_logits = model.decode_logits(batch, logits_all=len(batch) > 1)
        decode_batch_times.append(time.perf_counter() - t0d)
        passes += 1
        
        # VERIFY
        t0v = time.perf_counter()
        preds = np.empty(len(batch) + 1, dtype=np.intc)
        preds[0] = prev_pred
        if len(batch) > 1:
            preds[1:-1] = np.argmax(batch_logits[:-1], axis=1).astype(np.intc, copy=False)
        preds[-1] = qwen_mod.FastGreedyLlama.argmax(batch_logits[-1])
        
        accepted_batch = 0
        for i, tok in enumerate(batch):
            if int(preds[i]) == tok:
                accepted_batch += 1
            else:
                break
        bonus = int(preds[accepted_batch])
        verify_times.append(time.perf_counter() - t0v)
        
        rejected = accepted_batch < len(batch)
        
        # REJECT RECOVERY
        t0r = time.perf_counter()
        if rejected:
            if reject_mode == "seq-bonus":
                if accepted_batch > 0:
                    last_accepted_tok = batch[accepted_batch - 1]
                    model.truncate_kv(old_kv + accepted_batch - 1)
                    seq_logits = model.decode_logits([last_accepted_tok], logits_all=False)[0]
                    passes += 1
                    bonus = model.argmax(seq_logits)
                else:
                    model.truncate_kv(old_kv)
                    bonus = prev_pred
            elif reject_mode == "truncate":
                pass
        reject_times.append(time.perf_counter() - t0r)
        
        gen = gen[:old_kv] + batch[:accepted_batch] + [bonus]
        kv_len = old_kv + accepted_batch
        model.truncate_kv(kv_len)
        prev_pred = bonus
    
    elapsed = time.perf_counter() - start
    out_ids = gen[:target]
    t0 = time.perf_counter()
    _ = model.detokenize(out_ids[len(prompt_ids):])
    detok_time = time.perf_counter() - t0
    
    return {
        "passes": passes,
        "time": elapsed,
        "times": {
            "prompt_decode": prompt_decode_time,
            "pattern": pattern_times,
            "decode_batch": decode_batch_times,
            "verify": verify_times,
            "reject_recover": reject_times,
            "detok": detok_time,
        }
    }

# ── Compare ────────────────────────────────────────────────────────────
CORPUS = r"C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json"
MODEL = r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

print("=" * 80)
print(" DETAILATED PHASE BREAKDOWN — why spec is slower/faster")
print("=" * 80)

corpus = qwen_mod.QwenTokenCorpus(CORPUS)
miner = qwen_mod.PatternMiner(corpus.token_text, max_ctx=8, min_support=2, min_conf=0.96, det_conf=0.96, min_rule_ctx=4).fit(corpus.sequences)
extra = qwen_mod.extra_corpus_sequences(qwen_mod.FastGreedyLlama(MODEL))
qwen_mod.refit_miner(miner, list(corpus.sequences) + extra)
model = qwen_mod.FastGreedyLlama(MODEL, n_ctx=2048, n_gpu_layers=-1)
syntax = qwen_mod.PythonSyntaxProposer(model, mode="cluster")

prompts = qwen_mod.PROMPTS[:5]
results = []

for idx, prompt in enumerate(prompts, 1):
    gr = run_greedy_full_timing(model, prompt, 100)
    sp = run_spec_full_timing(model, miner, syntax, prompt, 100, k=6)
    results.append((gr, sp))
    
    print(f"\n[{idx:02d}] {prompt[:55]}")
    print(f"  Passes: greedy={gr['passes']}  spec={sp['passes']}  speed={gr['passes']/max(1,sp['passes']):.3f}×")
    print(f"  Total time: greedy={gr['time']:.3f}s  spec={sp['time']:.3f}s  speed={gr['time']/max(1e-9,sp['time']):.3f}×")
    print(f"  Phase breakdown (seconds):")
    print(f"    prompt_decode    {gr['times']['prompt_decode']:.4f}s  |  {sp['times']['prompt_decode']:.4f}s")
    print(f"    decode (∑batch)  {gr['times']['total_decode']:.4f}s  |  {sum(sp['times']['decode_batch']):.4f}s  (spec batch count={len(sp['times']['decode_batch'])})")
    print(f"    pattern          {'-':>6}    |  {sum(sp['times']['pattern']):.4f}s  (calls={len(sp['times']['pattern'])})")
    print(f"    verify           {'-':>6}    |  {sum(sp['times']['verify']):.4f}s")
    print(f"    reject_recover   {'-':>6}    |  {sum(sp['times']['reject_recover']):.4f}s")
    print(f"    detok            {gr['times']['detok']:.4f}s  |  {sp['times']['detok']:.4f}s")

# Aggregate
print("\n" + "=" * 80)
print("AGGREGATE PHASE TOTALS (5 prompts)")
print("=" * 80)
tot_g = {"prompt_decode":0, "total_decode":0, "detok":0}
tot_s = {"prompt_decode":0, "decode_batch":0, "pattern":0, "verify":0, "reject_recover":0, "detok":0}
for gr, sp in results:
    tot_g["prompt_decode"] += gr["times"]["prompt_decode"]
    tot_g["total_decode"] += gr["times"]["total_decode"]
    tot_g["detok"] += gr["times"]["detok"]
    tot_s["prompt_decode"] += sp["times"]["prompt_decode"]
    tot_s["decode_batch"] += sum(sp["times"]["decode_batch"])
    tot_s["pattern"] += sum(sp["times"]["pattern"])
    tot_s["verify"] += sum(sp["times"]["verify"])
    tot_s["reject_recover"] += sum(sp["times"]["reject_recover"])
    tot_s["detok"] += sp["times"]["detok"]

print(f"{'Phase':<25} {'Greedy (s)':<15} {'Spec (s)':<15} {'Notes'}")
print("-" * 80)
print(f"{'prompt_decode':<25} {tot_g['prompt_decode']:.4f}s       {tot_s['prompt_decode']:.4f}s  (prompt-only, single-pass)")
print(f"{'decode (generation)':<25} {tot_g['total_decode']:.4f}s       {tot_s['decode_batch']:.4f}s  ← main work")
print(f"{'pattern propose':<25} {'-':<15} {tot_s['pattern']:.4f}s  ({tot_s['pattern']/tot_s['decode_batch']*100:.1f}% of decode)")
print(f"{'verify':<25} {'-':<15} {tot_s['verify']:.4f}s  (tiny)")
print(f"{'reject_recover':<25} {'-':<15} {tot_s['reject_recover']:.4f}s  (seq-bonus extra eval)")
print(f"{'detok':<25} {tot_g['detok']:.4f}s       {tot_s['detok']:.4f}s")
print(f"\nTotal spec overhead beyond decode: {tot_s['pattern']+tot_s['verify']+tot_s['reject_recover']:.4f}s")
print(f"Total decode time: {tot_s['decode_batch']:.4f}s")
print(f"Spec extra overhead as % of greedy decode: { (tot_s['pattern']+tot_s['verify']+tot_s['reject_recover']) / max(1e-9, tot_g['total_decode']) * 100:.1f}%")
