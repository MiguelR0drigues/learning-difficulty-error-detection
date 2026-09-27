"""Diagnostic item bank for the screening mode (thesis Ch.4).

Design principle (the linguistic analogue of the math distinguishability
result): each item is chosen so that the ERROR IT ELICITS, if any, is
informative about WHICH difficulty category is implicated -- maximizing
expected information about the student's latent error profile per item,
instead of hoping free writing happens to exercise the right words.

Three linguistic item families, each targeting one taxonomy category:

- ORTHO-trap words: irregular/exception spellings whose pronunciation is
  recoverable from a wrong-but-phonetically-faithful spelling (e.g.
  'because' -> 'becaus', 'front' -> 'frunt'). A child with intact
  phonology but weak orthographic memory errs here WITH sound preserved
  (classified PHONO-faithful by the mapper) -- high error rate on this
  family with sound preserved signals the orthographic-memory profile.
- PHONO-transparent words: short, regular, fully decodable words ('stamp',
  'blend'). Errors here that ALTER the sound (voicing swaps, inversions,
  cluster reductions -> different Metaphone) cannot be blamed on English
  irregularity -- they signal the phoneme-grapheme-conversion profile
  associated with phonological deficits.
- SEG-compound items: compounds and multiword expressions with a plausible
  alternative segmentation ('sometimes' / 'some times', 'a lot' / 'alot').
  Fusion/splitting errors signal the word-boundary profile.

Math items reuse the distinguishability policy from
math/experiment_distinguishability.py: only problems where ALL modeled
malrules yield pairwise-distinct answers (and distinct from the correct
one) are admitted, so any malrule-matching wrong answer identifies its
misconception uniquely.
"""
from __future__ import annotations

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "math"))
from mal_rules import (all_subtraction_malrule_answers,  # noqa: E402
                        generate_subtraction_problem, generate_fraction,
                        correct_frac_add, malrule_frac_add)

# (word, hint-for-dictation) -- hint is what the teacher reads aloud / the
# picture shown; for the prototype the teacher dictates the word directly.
ORTHO_TRAP = ["because", "front", "would", "friend", "people", "said",
              "once", "again", "answer", "sure", "busy", "beautiful"]
PHONO_TRANSPARENT = ["stamp", "blend", "drink", "plant", "frost", "spot",
                     "milk", "hand", "crisp", "swim", "grab", "twist"]
SEG_COMPOUND = ["sometimes", "football", "birthday", "playground",
                "inside", "homework", "a lot", "ice cream", "outside",
                "everyone"]

LING_FAMILIES = {"ORTHO_TRAP": ORTHO_TRAP,
                 "PHONO_TRANSPARENT": PHONO_TRANSPARENT,
                 "SEG_COMPOUND": SEG_COMPOUND}

# ---------------------------------------------------------------------------
# European Portuguese instantiation (thesis Ch.4, Portuguese screening port).
# Same three families, same diagnostic logic; only the language resources
# change -- which is precisely the claim the thesis makes about the
# screening mode's portability (no corpus, no training, no neural model).
#
# ORTHO_TRAP_PT: words whose conventional spelling is NOT recoverable from
#   sound alone -- silent h (hoje), c/ç/ss ambiguity (cabeça), ch/x (chave),
#   g/j before e,i (girafa), intervocalic s=/z/ (casa). A phonetically-
#   faithful wrong form (oje, cabessa, xave, jirafa, caza) preserves the
#   sound -> ERR-PHONO route -> orthographic-memory signal, as in English.
# PHONO_TRANSPARENT_PT: short, fully decodable words with regular
#   letter-sound mapping (pato, bola, salada). Any deviation that ALTERS
#   the sound (voicing swaps p/b t/d f/v, cluster reductions pr->p,
#   letter inversions) cannot be blamed on Portuguese orthographic
#   irregularity -> phoneme-grapheme-conversion signal.
# SEG_COMPOUND_PT: items with a plausible alternative segmentation, chosen
#   around attested child fusions/splits (derrepente, asvezes, porisso,
#   concerteza are classic Portuguese classroom errors).
# ---------------------------------------------------------------------------
ORTHO_TRAP_PT = ["hoje", "hora", "homem", "chave", "chuva", "peixe",
                 "girafa", "gelado", "casa", "cabeça", "professor",
                 "também"]
PHONO_TRANSPARENT_PT = ["pato", "bola", "dedo", "faca", "sopa", "mala",
                        "fita", "sapo", "lua", "prato", "verde", "salada"]
SEG_COMPOUND_PT = ["de repente", "às vezes", "por isso", "com certeza",
                   "fim de semana", "em cima", "passatempo", "girassol",
                   "guarda-chuva", "afinal"]

LING_FAMILIES_PT = {"ORTHO_TRAP": ORTHO_TRAP_PT,
                    "PHONO_TRANSPARENT": PHONO_TRANSPARENT_PT,
                    "SEG_COMPOUND": SEG_COMPOUND_PT}

BANKS = {"en": LING_FAMILIES, "pt": LING_FAMILIES_PT}


def diagnostic_subtraction(rng: random.Random, digits: int = 5,
                            max_tries: int = 500) -> tuple[int, int]:
    """A subtraction problem on which every modeled malrule is uniquely
    identifiable from the answer (the >=4-digit + zero-borrow regime the
    distinguishability experiment identified)."""
    for _ in range(max_tries):
        a, b = generate_subtraction_problem(digits=digits, force_borrow=True,
                                            force_zero_borrow=True, rng=rng)
        ans = all_subtraction_malrule_answers(a, b)
        mal = {k: v for k, v in ans.items() if k != "CORRECT"}
        if len(mal) == 3 and len(set(mal.values())) == 3 and ans["CORRECT"] not in mal.values():
            return a, b
    raise RuntimeError("no fully-diagnostic problem found")


def diagnostic_fraction(rng: random.Random, max_tries: int = 200) -> tuple:
    """A fraction-addition item where the FRAC_ADD malrule answer differs
    from the correct answer (always true unless degenerate)."""
    for _ in range(max_tries):
        n1, d1 = generate_fraction(rng=rng)
        n2, d2 = generate_fraction(rng=rng)
        if correct_frac_add(n1, d1, n2, d2) != malrule_frac_add(n1, d1, n2, d2):
            return n1, d1, n2, d2
    raise RuntimeError("no diagnostic fraction found")


def build_worksheet(seed: int = 0, n_per_family: int = 4, n_sub: int = 3,
                     n_frac: int = 2, lang: str = "en") -> dict:
    """A screening worksheet: dictation words (mixed order, family recorded)
    + fully-diagnostic math items. `lang` selects the item bank ("en"/"pt");
    the math items are language-independent by construction."""
    rng = random.Random(seed)
    words = []
    for fam, bank in BANKS[lang].items():
        for w in rng.sample(bank, n_per_family):
            words.append({"target": w, "family": fam})
    rng.shuffle(words)
    subs = [diagnostic_subtraction(rng) for _ in range(n_sub)]
    fracs = [diagnostic_fraction(rng) for _ in range(n_frac)]
    return {"seed": seed, "lang": lang,
            "dictation": words,
            "subtraction": [{"a": a, "b": b} for a, b in subs],
            "fractions": [{"n1": n1, "d1": d1, "n2": n2, "d2": d2}
                           for n1, d1, n2, d2 in fracs]}


if __name__ == "__main__":
    for lang in ("en", "pt"):
        ws = build_worksheet(seed=42, lang=lang)
        print(f"[{lang}] Worksheet: {len(ws['dictation'])} dictation items, "
              f"{len(ws['subtraction'])} subtractions, {len(ws['fractions'])} fractions")
        for it in ws["dictation"]:
            print(f"  [{it['family']:<18}] {it['target']}")
        for s in ws["subtraction"]:
            print(f"  [SUB fully-diagnostic] {s['a']} - {s['b']}")
        for f in ws["fractions"]:
            print(f"  [FRAC] {f['n1']}/{f['d1']} + {f['n2']}/{f['d2']}")
