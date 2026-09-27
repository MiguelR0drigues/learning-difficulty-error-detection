"""Builds full-sentence, word-level BIO-tagged examples from real BEA-2019
M2 gold data, using taxonomy_map.py to decide which edits are in-scope.

Note: a sentence can contain multiple edits; only those that map to
ERR-PHONO/ERR-ORTHO/ERR-SEG are tagged, everything else (out-of-scope
errors per the thesis's own stated scope) is left as O. This is a
deliberate, documented simplification -- those tokens are still
"errorful" in a general GEC sense, just outside this thesis's taxonomy.
"""
from __future__ import annotations

import json
from m2_parser import parse_m2_file, M2Sentence
from taxonomy_map import map_edit, THESIS_LABELS


def sentence_to_bio(sent: M2Sentence) -> dict:
    tokens = list(sent.tokens)
    tags = ["O"] * len(tokens)
    any_in_scope = False
    for edit in sent.edits:
        o_str = " ".join(tokens[edit.start:edit.end])
        c_str = edit.replacement
        if not o_str or not c_str:
            continue
        mapped = map_edit(o_str, c_str, edit.error_type)
        if mapped.thesis_label == "OTHER":
            continue
        label = mapped.thesis_label.replace("ERR-", "")
        span = range(edit.start, edit.end)
        if not span:
            continue
        for i, idx in enumerate(span):
            if idx >= len(tags):
                continue
            tags[idx] = f"B-{label}" if i == 0 else f"I-{label}"
        any_in_scope = True
    return {"tokens": tokens, "tags": tags, "has_in_scope_error": any_in_scope}


def build_corpus(files: list) -> list:
    out = []
    for fp in files:
        for sent in parse_m2_file(fp):
            if not sent.edits:
                continue  # skip error-free sentences for the labeled corpus
            row = sentence_to_bio(sent)
            if row["has_in_scope_error"]:
                out.append(row)
    return out


if __name__ == "__main__":
    import os
    from collections import Counter

    # BEA-2019 W&I+LOCNESS (see data_strategy_plan.md for download info)
    bea_files = [
        "../data/linguistic_real/wi+locness/m2/A.train.gold.bea19.m2",
        "../data/linguistic_real/wi+locness/m2/B.train.gold.bea19.m2",
        "../data/linguistic_real/wi+locness/m2/C.train.gold.bea19.m2",
        "../data/linguistic_real/wi+locness/m2/N.dev.gold.bea19.m2",
    ]
    # FCE v2.1 (direct download, no form -- non-commercial research licence,
    # cite Yannakoudakis et al. 2011):
    #   https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
    # Extract so that fce/m2/*.m2 lands under data/linguistic_real/.
    fce_train_files = [
        "../data/linguistic_real/fce/m2/fce.train.gold.bea19.m2",
        "../data/linguistic_real/fce/m2/fce.dev.gold.bea19.m2",
    ]
    fce_test_file = "../data/linguistic_real/fce/m2/fce.test.gold.bea19.m2"

    def report(name, corpus):
        tag_counts = Counter(t for row in corpus for t in row["tags"] if t != "O")
        print(f"{name}: {len(corpus)} in-scope sentences | tags: {dict(tag_counts)}")

    bea = build_corpus([f for f in bea_files if os.path.exists(f)])
    report("BEA-2019", bea)
    with open("../data/linguistic/bea2019_bio_real.json", "w") as f:
        json.dump(bea, f, indent=1)

    fce_avail = [f for f in fce_train_files if os.path.exists(f)]
    if fce_avail:
        fce = build_corpus(fce_avail)
        report("FCE train+dev", fce)
        combined = bea + fce
        with open("../data/linguistic/real_bio_combined.json", "w") as f:
            json.dump(combined, f, indent=1)
        print(f"Combined real corpus: {len(combined)} -> data/linguistic/real_bio_combined.json")
        if os.path.exists(fce_test_file):
            fce_test = build_corpus([fce_test_file])
            report("FCE test (held-out benchmark, NOT for training)", fce_test)
            with open("../data/linguistic/fce_test_bio.json", "w") as f:
                json.dump(fce_test, f, indent=1)
    else:
        print("FCE m2 files not found -- combined corpus not rebuilt (BEA-only output written).")
