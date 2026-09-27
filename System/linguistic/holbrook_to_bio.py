"""Converts the Holbrook corpus (real writings of struggling secondary-school
children, from David Holbrook's 'English for the Rejected', 1964; tagged
version distributed by Roger Mitton, https://titan.dcs.bbk.ac.uk/~roger/corpora.html)
into (a) BIO-tagged sentences under the thesis taxonomy and (b) per-student
error profiles.

Why this corpus matters for the thesis (vs BEA/FCE):
- Domain match: children with literacy difficulties, not adult L2 learners --
  this is the closest public English data to the thesis's target population,
  so it serves as a DOMAIN-SHIFT EVALUATION set for the linguistic model
  (train on L2 GEC, test on struggling-children writing).
- Longitudinal-ish: each child (fictitious name) has multiple passages, so it
  also yields real per-student error-type distributions to ground the
  profiling module's archetypes (previously purely assumed/synthetic).

It is deliberately NOT added to the training pool.

Mapping logic: Holbrook has no ERRANT types, so we route each error pair
ourselves before reusing taxonomy_map.map_edit:
- same letters, different word boundaries -> map_edit handles (ERR-SEG);
- misspelling is NOT an English word -> treat as R:SPELL (map_edit then
  splits ERR-PHONO vs ERR-ORTHO by Metaphone match);
- misspelling IS a real English word -> real-word error: in-scope only if
  it's a curated homophone confusion (ERR-ORTHO), else OTHER (grammar etc.,
  e.g. 'go' for 'goes' -- out of the thesis's linguistic scope).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from taxonomy_map import map_edit, _normalize

TAGGED_PATH = "../data/linguistic_real/holbrook/holbrook-tagged.dat"
WORDLIST_PATH = "../data/linguistic_real/holbrook/words_alpha.txt"

ERR_RE = re.compile(r"<ERR\s+targ=([^>]*)>\s*(.*?)\s*</ERR>")
NAME_RE = re.compile(r"^([A-Z][A-Z' ]+?)\s+page\s+\d+\s*$")


def load_wordlist(path: str) -> set:
    return {w.strip().lower() for w in open(path) if w.strip()}


def classify(wrong: str, target: str, words: set) -> str:
    """Returns a thesis label for one Holbrook error pair."""
    if target.strip() in ("?", ""):
        return "OTHER"
    norm_w = _normalize(wrong)
    if not norm_w:
        return "OTHER"
    # pick pseudo-ERRANT type: spelling if the written form isn't a word
    wrong_toks = [_normalize(t) for t in wrong.split()]
    all_real_words = all(t in words for t in wrong_toks if t)
    etype = "R:OTHER" if all_real_words else "R:SPELL"
    return map_edit(wrong, target, etype).thesis_label


def parse(tagged_path: str, words: set):
    student = None
    rows = []
    for raw in open(tagged_path, encoding="utf-8", errors="replace"):
        line = raw.strip()
        if not line or re.fullmatch(r"\d+\.", line):
            continue
        m = NAME_RE.match(line)
        if m:
            student = m.group(1).title()
            continue
        # skip passage titles / markers (all-caps lines with no tagged errors)
        if line.upper() == line and not ERR_RE.search(line):
            continue
        if student is None:
            continue
        tokens, tags, labels_in_sentence = [], [], []
        pos = 0
        for em in ERR_RE.finditer(line):
            # plain text before the error
            for t in line[pos:em.start()].split():
                tokens.append(t); tags.append("O")
            wrong, target = em.group(2), em.group(1)
            label = classify(wrong, target, words)
            labels_in_sentence.append(label)
            span_toks = wrong.split() or [wrong]
            for i, t in enumerate(span_toks):
                tokens.append(t)
                if label == "OTHER":
                    tags.append("O")
                else:
                    short = label.replace("ERR-", "")
                    tags.append(f"B-{short}" if i == 0 else f"I-{short}")
            pos = em.end()
        for t in line[pos:].split():
            tokens.append(t); tags.append("O")
        if tokens:
            rows.append({"student": student, "tokens": tokens, "tags": tags,
                         "error_labels": labels_in_sentence})
    return rows


if __name__ == "__main__":
    words = load_wordlist(WORDLIST_PATH)
    rows = parse(TAGGED_PATH, words)

    # (a) BIO benchmark: sentences with >=1 in-scope tag
    bio = [{"tokens": r["tokens"], "tags": r["tags"]} for r in rows
           if any(t != "O" for t in r["tags"])]
    with open("../data/linguistic/holbrook_bio.json", "w") as f:
        json.dump(bio, f, indent=1)

    # (b) per-student error profiles (real data for the profiling module)
    profiles = defaultdict(lambda: {"ERR-PHONO": 0, "ERR-ORTHO": 0, "ERR-SEG": 0,
                                     "OTHER": 0, "n_sentences": 0, "n_tokens": 0})
    for r in rows:
        p = profiles[r["student"]]
        p["n_sentences"] += 1
        p["n_tokens"] += len(r["tokens"])
        for lab in r["error_labels"]:
            p[lab] += 1
    with open("../data/profiling/holbrook_student_profiles.json", "w") as f:
        json.dump(dict(profiles), f, indent=1)

    total_labels = Counter(l for r in rows for l in r["error_labels"])
    print(f"Parsed {len(rows)} sentences from {len(profiles)} students")
    print(f"Error label distribution: {dict(total_labels)}")
    print(f"BIO benchmark sentences (>=1 in-scope error): {len(bio)}")
    for s, p in sorted(profiles.items()):
        total_err = p["ERR-PHONO"] + p["ERR-ORTHO"] + p["ERR-SEG"]
        print(f"  {s:<20} sents={p['n_sentences']:>3} PHONO={p['ERR-PHONO']:>3} "
              f"ORTHO={p['ERR-ORTHO']:>3} SEG={p['ERR-SEG']:>2} OTHER={p['OTHER']:>3}")
