"""Closed-world (known-target) deterministic scorer for screening items.

Because the target is known, error classification requires no model
inference and inherits no domain gap: the written form is aligned with the
target and routed through the same taxonomy logic validated against real
corpus data (linguistic/taxonomy_map.py + the real-word routing rule from
linguistic/holbrook_to_bio.py).

Output per item: CORRECT | ERR-PHONO | ERR-ORTHO | ERR-SEG | OTHER.
Aggregation: per-family error rates + z-scores against a class norm.

Languages: English (Metaphone comparator, taxonomy_map) and European
Portuguese (pt_g2p comparator) -- same decision structure in both.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "linguistic"))
from taxonomy_map import map_edit, _normalize  # noqa: E402

from pt_g2p import pt_phonemes  # noqa: E402

_WORDS = None


def _wordlist() -> set:
    global _WORDS
    if _WORDS is None:
        path = os.path.join(os.path.dirname(__file__), "..",
                            "data", "linguistic_real", "holbrook", "words_alpha.txt")
        _WORDS = {w.strip().lower() for w in open(path) if w.strip()}
    return _WORDS


def classify_item(written: str, target: str) -> str:
    """Deterministic classification of a screening response (English)."""
    written = written.strip()
    if not written:
        return "OTHER"
    if _normalize(written) == _normalize(target) and \
       written.count(" ") == target.count(" "):
        return "CORRECT"
    # Closed-world routing differs deliberately from the free-text rule
    # used for Holbrook: there, a real-word form ('go' for 'goes') is
    # likely grammatical and must be excluded; in dictation the target is
    # fixed, so ANY deviation is a spelling attempt at that target -- even
    # if it happens to be a real word (e.g. 'blent' for 'blend', a genuine
    # voicing error that the free-text rule would wrongly discard). All
    # non-SEG deviations therefore route through the R:SPELL path
    # (Metaphone split into PHONO vs ORTHO). Known granularity limit:
    # Metaphone cannot see through some letter transpositions
    # ('becuase'/'because' share a code and land PHONO).
    return map_edit(written, target, "R:SPELL").thesis_label


# ---------------------------------------------------------------------------
# Portuguese (EP) closed-world classification. Same decision structure as
# the English path -- CORRECT / ERR-SEG / sound-preserved ERR-PHONO /
# sound-altering ERR-ORTHO -- with pt_g2p as the phonetic comparator in
# place of Metaphone.
# ---------------------------------------------------------------------------

def _pt_letters(s: str) -> str:
    """Lowercased letters only, accents and ç preserved (correctness must
    be accent-sensitive: 'agua' for 'água' is an error, not CORRECT)."""
    return "".join(ch for ch in s.lower() if ch.isalpha())


def _pt_boundaries(s: str) -> str:
    """Sequence of word-boundary markers, normalized: each boundary is
    '-' if it contains a hyphen, else ' '. Both the number AND the kind of
    boundary must match: 'guarda chuva' for 'guarda-chuva' keeps the count
    but changes the marker, and is still a boundary (SEG) error."""
    return "".join("-" if "-" in m else " "
                   for m in re.findall(r"[\s\-]+", s.strip()))


def _pt_seg_norm(s: str) -> str:
    """Normal form for the segmentation test: does the written form contain
    the same word-material as the target, just re-segmented? Strips accents
    and boundaries, then collapses the letter alternations that Portuguese
    orthography FORCES on a child who merges or splits words: rr/r and ss/s
    (fusing 'de repente' correctly by sound yields 'derrepente'), and
    coda m/n (fusing 'com certeza' yields 'concerteza')."""
    base = "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if not unicodedata.combining(c))
    base = base.replace("ç", "c")
    base = re.sub(r"[\s\-]+", "", base)
    base = base.replace("rr", "r").replace("ss", "s").replace("m", "n")
    return "".join(ch for ch in base if ch.isalpha())


def _pt_sound(s: str) -> str:
    """Concatenated per-token phonemes, so multiword targets keep each
    token's word-initial/final phonology (strong R, final -e elision)."""
    return "".join(pt_phonemes(t) for t in re.split(r"[\s\-]+", s.strip()) if t)


def classify_item_pt(written: str, target: str) -> str:
    written = written.strip()
    if not written:
        return "OTHER"
    if _pt_letters(written) == _pt_letters(target) and \
       _pt_boundaries(written) == _pt_boundaries(target):
        return "CORRECT"
    if _pt_seg_norm(written) == _pt_seg_norm(target) and \
       _pt_boundaries(written) != _pt_boundaries(target):
        return "ERR-SEG"
    if _pt_sound(written) == _pt_sound(target):
        return "ERR-PHONO"   # sound preserved: phonetically-faithful form
    return "ERR-ORTHO"       # sound altered: visual / conversion slip


def score_worksheet(worksheet: dict, answers: dict) -> dict:
    """answers: {"dictation": [written...], "subtraction": [int...],
    "fractions": [Fraction-string...]} aligned with the worksheet order."""
    from fractions import Fraction
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "math"))
    from mal_rules import detect_subtraction_error, detect_fraction_add_error

    lang = worksheet.get("lang", "en")
    _classify = classify_item_pt if lang == "pt" else classify_item
    per_family = {}
    detail = []
    for item, written in zip(worksheet["dictation"], answers.get("dictation", [])):
        label = _classify(written, item["target"])
        detail.append({"target": item["target"], "written": written,
                       "family": item["family"], "label": label})
        fam = per_family.setdefault(item["family"], {"n": 0, "errors": 0})
        fam["n"] += 1
        if label != "CORRECT":
            fam["errors"] += 1
    math_labels = []
    for prob, ans in zip(worksheet["subtraction"], answers.get("subtraction", [])):
        try:
            math_labels.append(detect_subtraction_error(prob["a"], prob["b"], int(ans)))
        except (ValueError, TypeError):
            math_labels.append("INVALID")
    for prob, ans in zip(worksheet["fractions"], answers.get("fractions", [])):
        try:
            fr = Fraction(str(ans))
            math_labels.append(detect_fraction_add_error(prob["n1"], prob["d1"],
                                                          prob["n2"], prob["d2"], fr))
        except (ValueError, ZeroDivisionError):
            math_labels.append("INVALID")
    return {"dictation_detail": detail,
            "family_rates": {f: v["errors"] / v["n"] for f, v in per_family.items()},
            "math_labels": math_labels}


def flag_against_norm(family_rates: dict, class_rates: dict,
                       class_sd: dict, z_threshold: float = 1.5) -> list:
    """Returns families where the student deviates > z_threshold SDs above
    the class mean -- the screening 'needs attention' signal."""
    flags = []
    for fam, rate in family_rates.items():
        mu = class_rates.get(fam, 0.0)
        sd = max(class_sd.get(fam, 0.15), 0.05)
        z = (rate - mu) / sd
        if z >= z_threshold:
            flags.append({"family": fam, "rate": rate, "class_mean": mu,
                          "z": round(z, 2)})
    return flags


if __name__ == "__main__":
    # self-test with archetypal responses
    cases = [
        ("becaus", "because", "ERR-PHONO"),     # sound preserved -> phonetically-faithful
        ("frunt", "front", "ERR-PHONO"),
        ("becuase", "because", "ERR-PHONO"),    # transposition; Metaphone-equal -> PHONO (known granularity limit)
        ("blent", "blend", "ERR-PHONO"),        # voicing swap d->t: same metaphone? check
        ("some times", "sometimes", "ERR-SEG"),
        ("alot", "a lot", "ERR-SEG"),
        ("football", "football", "CORRECT"),
        ("would", "would", "CORRECT"),
    ]
    ok = 0
    for written, target, expect in cases:
        got = classify_item(written, target)
        mark = "OK " if got == expect else "?? "
        if got == expect:
            ok += 1
        print(f"{mark} {written!r} vs {target!r} -> {got} (expected {expect})")
    print(f"[en] {ok}/{len(cases)} as expected")

    # Portuguese (EP) archetypal responses -- one per designed trap pattern.
    cases_pt = [
        # ORTHO_TRAP items, phonetically-faithful forms -> ERR-PHONO
        ("oje", "hoje", "ERR-PHONO"),          # silent h dropped
        ("xave", "chave", "ERR-PHONO"),        # ch/x
        ("jirafa", "girafa", "ERR-PHONO"),     # g/j before i
        ("caza", "casa", "ERR-PHONO"),         # intervocalic s = /z/
        ("cabessa", "cabeça", "ERR-PHONO"),    # ç/ss
        ("tanbem", "também", "ERR-PHONO"),     # coda nasal + accent dropped
        # PHONO_TRANSPARENT items, sound-altering slips -> ERR-ORTHO
        ("bato", "pato", "ERR-ORTHO"),         # voicing swap p/b
        ("tedo", "dedo", "ERR-ORTHO"),         # voicing swap d/t
        ("pato", "prato", "ERR-ORTHO"),        # cluster reduction
        ("vedre", "verde", "ERR-ORTHO"),       # inversion
        # SEG_COMPOUND items -> ERR-SEG
        ("derrepente", "de repente", "ERR-SEG"),
        ("asvezes", "às vezes", "ERR-SEG"),
        ("porisso", "por isso", "ERR-SEG"),
        ("concerteza", "com certeza", "ERR-SEG"),
        ("passa tempo", "passatempo", "ERR-SEG"),
        ("guarda chuva", "guarda-chuva", "ERR-SEG"),
        # correctness is accent-sensitive
        ("agua", "água", "ERR-PHONO"),         # accent omission = error, sound kept
        ("hoje", "hoje", "CORRECT"),
        ("às vezes", "às vezes", "CORRECT"),
        ("cabeça", "cabeça", "CORRECT"),
    ]
    ok = 0
    for written, target, expect in cases_pt:
        got = classify_item_pt(written, target)
        mark = "OK " if got == expect else "?? "
        if got == expect:
            ok += 1
        print(f"{mark} [pt] {written!r} vs {target!r} -> {got} (expected {expect})")
    print(f"[pt] {ok}/{len(cases_pt)} as expected")
