"""Ch.5 experiment: how much does the open-world detection gap distort the
per-student PROFILES the system ultimately cares about?

Setup: for each of the 19 Holbrook students, build two error-type
distributions over {PHONO, ORTHO, SEG}:
  - closed-world: from the gold (target-known, deterministically mapped)
    labels -- what the screening scorer would produce;
  - open-world: from the neural model's free-text detections on the same
    sentences (argmax decoding, as in the Ch.5 benchmarks).

Metrics per student: cosine similarity between the two 3-dim proportion
vectors; plus, across students, Spearman rank correlation per category
(does the model still RANK students correctly on each error type, even if
absolute proportions are biased?) -- ranking is what a teacher's
attention-allocation actually uses.

Chunked CLI (CPU inference is slow in the sandbox):
  python experiment_profile_fidelity.py --start 0 --end 300
  ... (repeat) ...
  python experiment_profile_fidelity.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "linguistic"))

PARTIAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiling", "fidelity_partials")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "profiling", "profile_fidelity_results.json")
CATS = ["PHONO", "ORTHO", "SEG"]


def load_rows():
    from holbrook_to_bio import parse, load_wordlist, TAGGED_PATH, WORDLIST_PATH
    base = os.path.join(os.path.dirname(__file__), "..", "linguistic")
    words = load_wordlist(os.path.join(base, WORDLIST_PATH))
    return parse(os.path.join(base, TAGGED_PATH), words)


def spans_from_tags(tags: list) -> list:
    out, cur = [], None
    for t in tags:
        if t.startswith("B-"):
            cur = t[2:]
            out.append(cur)
        elif not t.startswith("I-"):
            cur = None
    return out


def run_chunk(start: int, end: int):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    rows = load_rows()[start:end]
    model_dir = os.path.join(os.path.dirname(__file__), "..", "linguistic", "final_model")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    LABELS = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]
    results = []
    B = 16
    for i in range(0, len(rows), B):
        batch = rows[i:i + B]
        enc = tok([r["tokens"] for r in batch], is_split_into_words=True,
                  truncation=True, max_length=64, padding=True, return_tensors="pt")
        with torch.no_grad():
            preds = model(**enc).logits.argmax(dim=-1)
        for j, r in enumerate(batch):
            word_pred = {}
            for wid, p in zip(enc.word_ids(batch_index=j), preds[j].tolist()):
                if wid is not None and wid not in word_pred:
                    word_pred[wid] = LABELS[p]
            pred_tags = [word_pred.get(k, "O") for k in range(len(r["tokens"]))]
            results.append({"student": r["student"],
                            "gold_spans": spans_from_tags(r["tags"]),
                            "pred_spans": spans_from_tags(pred_tags)})
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    out = os.path.join(PARTIAL_DIR, f"part_{start}_{end}.json")
    json.dump(results, open(out, "w"))
    print(f"chunk [{start}:{end}] -> {len(results)} rows -> {out}")


def aggregate():
    import numpy as np
    rows = []
    for f in sorted(os.listdir(PARTIAL_DIR)):
        rows.extend(json.load(open(os.path.join(PARTIAL_DIR, f))))
    print(f"{len(rows)} scored sentences")
    gold = defaultdict(lambda: {c: 0 for c in CATS})
    pred = defaultdict(lambda: {c: 0 for c in CATS})
    for r in rows:
        for s in r["gold_spans"]:
            gold[r["student"]][s] += 1
        for s in r["pred_spans"]:
            pred[r["student"]][s] += 1
    students = sorted(set(gold) | set(pred))

    def props(d):
        tot = sum(d.values())
        return np.array([d[c] / tot for c in CATS]) if tot else np.zeros(3)

    cos_sims, table = [], []
    for st in students:
        g, p = props(gold[st]), props(pred[st])
        denom = (np.linalg.norm(g) * np.linalg.norm(p))
        cos = float(g @ p / denom) if denom else 0.0
        cos_sims.append(cos)
        table.append({"student": st, "gold_props": g.round(3).tolist(),
                      "pred_props": p.round(3).tolist(), "cosine": round(cos, 3),
                      "gold_total": int(sum(gold[st].values())),
                      "pred_total": int(sum(pred[st].values()))})
        print(f"  {st:<18} cos={cos:.3f} gold={g.round(2)} pred={p.round(2)}")

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        rx = rx - rx.mean(); ry = ry - ry.mean()
        d = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
        return float((rx * ry).sum() / d) if d else 0.0

    rank_corr = {}
    for k, c in enumerate(CATS):
        x = np.array([props(gold[st])[k] for st in students])
        y = np.array([props(pred[st])[k] for st in students])
        rank_corr[c] = round(spearman(x, y), 3)
    # density ranking: total errors per sentence
    sent_counts = defaultdict(int)
    for r in rows:
        sent_counts[r["student"]] += 1
    xg = np.array([sum(gold[st].values()) / sent_counts[st] for st in students])
    xp = np.array([sum(pred[st].values()) / sent_counts[st] for st in students])
    rank_corr["ERROR_DENSITY"] = round(spearman(xg, xp), 3)

    summary = {"n_students": len(students), "n_sentences": len(rows),
               "mean_cosine": round(float(np.mean(cos_sims)), 3),
               "min_cosine": round(float(np.min(cos_sims)), 3),
               "spearman_rank_correlation": rank_corr,
               "per_student": table}
    json.dump(summary, open(OUT_PATH, "w"), indent=1)
    print(f"\nmean cosine(profile_open, profile_closed) = {summary['mean_cosine']}")
    print(f"rank correlations: {rank_corr}")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run_chunk(args.start, args.end)
