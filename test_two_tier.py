"""Test two-tier rule system on best config (k=8, truncate)."""
import sys, time, gc
sys.path.insert(0, r"C:\Users\neera\OneDrive\Desktop\sep")
import qwen_dsa_pattern_spec_from_scratch as qwen_mod

MODEL_PATH = r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
CORPUS_PATH = r"C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json"

print("=" * 80)
print("TWO-TIER RULE SYSTEM TEST — k=8, truncate")
print("=" * 80)

corpus = qwen_mod.QwenTokenCorpus(CORPUS_PATH)
miner = qwen_mod.PatternMiner(
    corpus.token_text, max_ctx=8,
    min_support=1, min_conf=0.88, det_conf=0.92, min_rule_ctx=4,
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
        k=8, reject_mode="truncate",
    )
    ok += int(greedy_ids == spec_ids)
    pg += gm["passes"]
    ps += sm["passes"]
    tg += gm["time"]
    ts += sm["time"]
    all_rec.extend(sm.get("records", []))
    if idx % 5 == 0:
        print(f"  [{idx:02d}/20] passes: {pg}/{ps} = {pg/max(1,ps):.3f}x  matches: {ok}/{idx}")

cov = sum(1 for r in all_rec if r["draft"] > 0) / len(all_rec) if all_rec else 0.0
prop = sum(int(r["draft"]) for r in all_rec)
acc = sum(int(r["accepted_draft"]) for r in all_rec)

print(f"\nPass speedup: {pg/max(1,ps):.3f}x")
print(f"Time speedup: {tg/max(1e-9,ts):.3f}x")
print(f"Match rate: {ok/20*100:.0f}%")
print(f"Coverage: {cov*100:.1f}%")
print(f"Accuracy: {acc/max(1,prop)*100:.1f}%")
print(f"Total wall time: {ts:.1f}s")
