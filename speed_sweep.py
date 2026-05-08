"""Focused config sweep — test highest-impact variants around current best."""
import sys, time, gc, json
sys.path.insert(0, r"C:\Users\neera\OneDrive\Desktop\sep")
import qwen_dsa_pattern_spec_from_scratch as qwen_mod

MODEL_PATH = r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
CORPUS_PATH = r"C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json"

# Configs to test (focused around current best)
CONFIGS = [
    {"label": "control (current best)", "min_support": 1, "min_conf": 0.88, "det_conf": 0.92, "k": 6, "reject_mode": "truncate"},
    {"label": "k=4 truncate", "min_support": 1, "min_conf": 0.88, "det_conf": 0.92, "k": 4, "reject_mode": "truncate"},
    {"label": "k=8 truncate", "min_support": 1, "min_conf": 0.88, "det_conf": 0.92, "k": 8, "reject_mode": "truncate"},
    {"label": "k=6 seq-bonus", "min_support": 1, "min_conf": 0.88, "det_conf": 0.92, "k": 6, "reject_mode": "seq-bonus"},
    {"label": "tighter min_conf=0.90", "min_support": 1, "min_conf": 0.90, "det_conf": 0.92, "k": 6, "reject_mode": "truncate"},
    {"label": "looser det_conf=0.90", "min_support": 1, "min_conf": 0.88, "det_conf": 0.90, "k": 6, "reject_mode": "truncate"},
    {"label": "strict support=2", "min_support": 2, "min_conf": 0.88, "det_conf": 0.92, "k": 6, "reject_mode": "truncate"},
]

def run_one(cfg):
    corpus = qwen_mod.QwenTokenCorpus(CORPUS_PATH)
    miner = qwen_mod.PatternMiner(
        corpus.token_text, max_ctx=8,
        min_support=cfg["min_support"],
        min_conf=cfg["min_conf"],
        det_conf=cfg["det_conf"],
        min_rule_ctx=4,
    ).fit(corpus.sequences)
    model = qwen_mod.FastGreedyLlama(MODEL_PATH, n_ctx=2048, n_gpu_layers=-1)
    syntax = qwen_mod.PythonSyntaxProposer(model, mode="cluster")
    prompts = qwen_mod.PROMPTS[:20]

    ok = pg = ps = tg = ts = 0
    all_rec = []

    for idx, prompt in enumerate(prompts, 1):
        greedy_ids, _, gm = qwen_mod.run_greedy(model, prompt, 100)
        spec_ids, _, sm = qwen_mod.run_speculative(
            model, miner, syntax, prompt, 100,
            k=cfg["k"], reject_mode=cfg["reject_mode"],
        )
        ok += int(greedy_ids == spec_ids)
        pg += gm["passes"]
        ps += sm["passes"]
        tg += gm["time"]
        ts += sm["time"]
        all_rec.extend(sm.get("records", []))
        if idx % 5 == 0:
            print(f"    [{idx:02d}/20] passes: {pg}/{ps} = {pg/max(1,ps):.3f}x  matches: {ok}/{idx}")

    cov = sum(1 for r in all_rec if r["draft"] > 0) / len(all_rec) if all_rec else 0.0
    prop = sum(int(r["draft"]) for r in all_rec)
    acc = sum(int(r["accepted_draft"]) for r in all_rec)

    rules_count = len(miner.rules_by_ctx)
    del miner, corpus, model, syntax
    gc.collect()

    return {
        "pass_speedup": pg / max(1, ps),
        "time_speedup": tg / max(1e-9, ts),
        "match_rate": ok / 20,
        "coverage": cov,
        "accuracy": acc / max(1, prop),
        "rules": rules_count,
        "wall_time": ts,
    }

print("=" * 80)
print("FOCUSED CONFIG SWEEP — 7 variants around current best")
print("=" * 80)

results = []
for i, cfg in enumerate(CONFIGS, 1):
    print(f"\n[{i}/{len(CONFIGS)}] {cfg['label']}")
    t0 = time.perf_counter()
    r = run_one(cfg)
    t1 = time.perf_counter()
    results.append((cfg, r))
    print(f"  Pass: {r['pass_speedup']:.3f}x  Time: {r['time_speedup']:.3f}x  Match: {r['match_rate']*100:.0f}%")
    print(f"  Cover: {r['coverage']*100:.1f}%  Acc: {r['accuracy']*100:.1f}%  Rules: {r['rules']}")
    print(f"  Sweep runtime: {t1-t0:.1f}s")

print("\n" + "=" * 80)
print("SUMMARY — sorted by wall time speedup")
print("=" * 80)
print(f"{'Config':<28} {'Pass↑':>8} {'Time↑':>8} {'Match':>6} {'Cover':>6} {'Acc':>6} {'Rules':>8}")
print("-" * 80)
for cfg, r in sorted(results, key=lambda x: -x[1]["time_speedup"]):
    print(f"{cfg['label']:<28} {r['pass_speedup']:>7.3f}x {r['time_speedup']:>7.3f}x {r['match_rate']*100:>5.0f}% {r['coverage']*100:>5.1f}% {r['accuracy']*100:>5.1f}% {r['rules']:>8}")

best = max(results, key=lambda x: x[1]["time_speedup"])
print(f"\nBEST CONFIG: {best[0]['label']}")
print(f"  Time speedup: {best[1]['time_speedup']:.3f}x")
