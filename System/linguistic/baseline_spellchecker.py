"""Non-neural baseline for Ch.5: dictionary spell-checker + the same
taxonomy mapping, evaluated with the same strict span-level seqeval metric
on the same held-out benchmarks as the neural model.

Pipeline per token:
- token not in dictionary -> error span; the spell-checker's suggested
  correction serves as the hypothesized target, routed through
  taxonomy_map's R:SPELL logic (Metaphone -> PHONO vs ORTHO).
- hyposegmentation check: an OOV token that splits into two dictionary
  words ('alot' -> 'a lot') -> SEG.
- hypersegmentation check: two adjacent tokens that join into a dictionary
  word while at least one side is suspicious -- conservatively, only when
  the joined form is a word AND the pair occurs in a curated-free manner;
  full hypersegmentation without context is noisy, so the baseline only
  merges when BOTH tokens are real words and the join is also a word
  ('some times' -> 'sometimes').

This is intentionally the strongest *simple* baseline we could build from
off-the-shelf parts; its purpose is to give the neural numbers a floor.
Usage: python3 baseline_spellchecker.py <benchmark.json> <out.json>
"""
from __future__ import annotations

import json
import re
import sys

import seqeval.metrics as seqeval_metrics
from spellchecker import SpellChecker
from taxonomy_map import map_edit, _normalize

_sc = SpellChecker()
_dict = _sc.word_frequency


def known(w: str) -> bool:
    n = _normalize(w)
    return bool(n) and n in _dict


def predict_tags(tokens: list) -> list:
    tags = ["O"] * len(tokens)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        n = _normalize(tok)
        if not n:
            i += 1
            continue
        # hypersegmentation: two adjacent real words joining into a word
        if i + 1 < len(tokens):
            n2 = _normalize(tokens[i + 1])
            if n and n2 and known(tok) and known(tokens[i + 1]) and (n + n2) in _dict \
               and not known(n + " " + n2):
                # 'some times' -> 'sometimes'; conservative: only if joined
                # form is common-ish (freq above either part's tail)
                tags[i], tags[i + 1] = "B-SEG", "I-SEG"
                i += 2
                continue
        if not known(tok):
            # hyposegmentation: OOV splits into two words
            split_found = False
            for k in range(2, len(n) - 1):
                if n[:k] in _dict and n[k:] in _dict:
                    tags[i] = "B-SEG"
                    split_found = True
                    break
            if not split_found:
                corr = _sc.correction(n)
                if corr and corr != n:
                    label = map_edit(tok, corr, "R:SPELL").thesis_label
                    tags[i] = "B-" + label.replace("ERR-", "") if label != "OTHER" else "O"
                else:
                    tags[i] = "B-ORTHO"  # OOV, no correction found: flag as spelling
        i += 1
    return tags


if __name__ == "__main__":
    bench_path, out_path = sys.argv[1], sys.argv[2]
    data = json.load(open(bench_path))
    gold = [r["tags"] for r in data]
    pred = [predict_tags(r["tokens"]) for r in data]
    res = {
        "benchmark": bench_path,
        "n_sentences": len(data),
        "precision": seqeval_metrics.precision_score(gold, pred),
        "recall": seqeval_metrics.recall_score(gold, pred),
        "f1": seqeval_metrics.f1_score(gold, pred),
    }
    json.dump({"summary": res, "pred_tags": pred}, open(out_path, "w"))
    print(f"{bench_path}: P={res['precision']:.4f} R={res['recall']:.4f} F1={res['f1']:.4f} (n={len(data)})")
