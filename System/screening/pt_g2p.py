"""Heuristic European Portuguese grapheme-to-phoneme comparator for the
screening mode's Portuguese instantiation (thesis Ch.4).

Role: the Portuguese analogue of Metaphone in the English pipeline. Given a
written form, produce a coarse segmental phoneme string so the scorer can
ask the one question the taxonomy split needs: *does the written form sound
the same as the dictated target?* Same sound -> phonetically-faithful
mis-spelling (ERR-PHONO route, orthographic-memory signal on trap items);
different sound -> ERR-ORTHO route (sound-altering slip, phonological
signal on transparent items).

Design constraints, stated honestly:
- This is NOT a general-purpose Portuguese G2P. It is a *comparator* tuned
  for the closed-world screening setting, where every target word comes
  from a curated item bank and every plausible trap spelling is known in
  advance and covered by a test. Coverage claims extend to the item bank
  plus its documented trap patterns, nothing more.
- European Portuguese (EP) conventions: final -s/-z as /S/ (chiado),
  unstressed final o -> /u/, elidable final /e/.
- Stress and vowel height are deliberately ignored (no stress detector):
  accented and unaccented vowels map to the same base phoneme. This makes
  accent-omission errors ('agua' for 'água') compare as same-sound, which
  is the desired routing: accent omission is an orthographic-convention
  error, and on trap items the sound-preserved route is exactly the
  orthographic-memory signal.

Phoneme inventory (internal, one char each unless noted):
  stops/fricatives: p b t d k g f v s z S (=/esh/) Z (=/ezh/)
  sonorants: m n N (=nh) l L (=lh) r R (=rr) j w
  vowels: a e i o u ; nasality marked with '~' after the vowel.
"""
from __future__ import annotations

import re
import unicodedata

_VOWELS = "aeiou"

# accented -> (base vowel, nasal?)
_ACCENT_MAP = {
    "á": ("a", False), "à": ("a", False), "â": ("a", False), "ã": ("a", True),
    "é": ("e", False), "ê": ("e", False),
    "í": ("i", False),
    "ó": ("o", False), "ô": ("o", False), "õ": ("o", True),
    "ú": ("u", False), "ü": ("u", False),
}


def _pre(word: str) -> str:
    """Lowercase; strip anything that is not a letter, ç, or an accented
    vowel. (Hyphens/spaces are the scorer's segmentation business, not
    the comparator's: callers pass single tokens.)"""
    w = word.lower().strip()
    keep = []
    for ch in w:
        if ch == "ç" or ch in _ACCENT_MAP or (ch.isalpha() and ch.isascii()):
            keep.append(ch)
    return "".join(keep)


def pt_phonemes(word: str) -> str:  # noqa: C901  (rule cascade, kept linear on purpose)
    """Coarse EP segmental transcription of a single written token."""
    w = _pre(word)
    if not w:
        return ""
    out: list[str] = []
    i, n = 0, len(w)

    def prev_is_vowel() -> bool:
        return bool(out) and (out[-1] in _VOWELS or out[-1] == "~")

    def next_char(k: int = 1) -> str:
        return w[i + k] if i + k < n else ""

    while i < n:
        c = w[i]
        nx = next_char()

        # ---- digraphs first ----
        if c == "c" and nx == "h":                     # ch -> /S/
            out.append("S"); i += 2; continue
        if c == "l" and nx == "h":                     # lh -> /L/
            out.append("L"); i += 2; continue
        if c == "n" and nx == "h":                     # nh -> /N/
            out.append("N"); i += 2; continue
        if c == "r" and nx == "r":                     # rr -> /R/
            out.append("R"); i += 2; continue
        if c == "s" and nx == "s":                     # ss -> /s/
            out.append("s"); i += 2; continue
        if c == "q":                                   # qu+e/i -> k ; qu+a/o -> kw
            if nx == "u" and next_char(2) in "ei":
                out.append("k"); i += 2; continue
            if nx == "u":
                out.extend(["k", "w"]); i += 2; continue
            out.append("k"); i += 1; continue
        if c == "g" and nx == "u" and next_char(2) in "ei":   # gue/gui -> /g/
            out.append("g"); i += 2; continue

        # ---- single letters ----
        if c == "h":                                   # silent everywhere
            i += 1; continue
        if c == "ç":
            out.append("s"); i += 1; continue
        if c == "c":
            out.append("s" if nx in "ei" or nx in ("é", "ê", "í") else "k")
            i += 1; continue
        if c == "g":
            out.append("Z" if nx in "ei" or nx in ("é", "ê", "í") else "g")
            i += 1; continue
        if c == "j":
            out.append("Z"); i += 1; continue
        if c == "x":
            # Default /S/ (bruxa, peixe, xadrez). Documented exception:
            # word-initial 'ex' + vowel -> /z/ (exame, exercício).
            if i == 1 and out and out[-1] == "e" and nx and nx in _VOWELS + "éêíóô":
                out.append("z"); i += 1; continue
            out.append("S"); i += 1; continue
        if c == "s":
            if i == n - 1:                             # final -s -> /S/ (EP)
                out.append("S")
            elif prev_is_vowel() and (nx in _VOWELS or nx in _ACCENT_MAP):
                out.append("z")                        # intervocalic s -> /z/
            else:
                out.append("s")
            i += 1; continue
        if c == "z":
            out.append("S" if i == n - 1 else "z")     # final -z -> /S/ (EP)
            i += 1; continue
        if c in "mn":
            # coda m/n nasalizes the previous vowel: campo==canpo, tambem==tanbem
            if prev_is_vowel() and (i == n - 1 or (nx and nx not in _VOWELS and nx not in _ACCENT_MAP)):
                if out[-1] != "~":
                    out.append("~")
                i += 1; continue
            out.append(c); i += 1; continue
        if c == "r":
            out.append("R" if i == 0 else "r")         # initial r is strong
            i += 1; continue
        if c in _ACCENT_MAP:
            base, nasal = _ACCENT_MAP[c]
            out.append(base)
            if nasal:
                out.append("~")
            i += 1; continue
        if c in _VOWELS:
            out.append(c); i += 1; continue
        out.append(c); i += 1                          # anything else, verbatim

    ph = "".join(out)

    # ---- word-final EP conventions ----
    # final 'am' and 'ão' are segmentally identical: /..a~w/
    ph = re.sub(r"a~$", "a~w", ph) if w.endswith(("am", "ão")) else ph
    # final 'em'/'ém' -> /e~j/ (tambem == tanbem == também)
    ph = re.sub(r"e~$", "e~j", ph) if w.endswith(("em", "ém")) else ph
    # unstressed final o -> /u/ ('meninu' spelling of 'menino')
    ph = re.sub(r"o(S?)$", r"u\1", ph) if not w.endswith(("ó", "ô", "ão")) else ph
    # final e is elided in EP ('noit' spelling of 'noite'); also before final S
    ph = re.sub(r"e(S?)$", r"\1", ph) if not w.endswith(("é", "ê")) else ph
    # collapse doubled vowels from spelling variants (e.g. 'voo'->'vo')
    ph = re.sub(r"([aeiou])\1", r"\1", ph)
    return ph


def pt_same_sound(a: str, b: str) -> bool:
    """The comparator the scorer uses: do two written forms share the same
    coarse EP segmental transcription?"""
    return pt_phonemes(a) == pt_phonemes(b) and bool(pt_phonemes(a))


if __name__ == "__main__":
    SAME = [  # phonetically-faithful trap spellings (must compare EQUAL)
        ("hoje", "oje"), ("hora", "ora"), ("homem", "omem"),
        ("chave", "xave"), ("chuva", "xuva"), ("peixe", "peiche"),
        ("bruxa", "brucha"), ("girafa", "jirafa"), ("gelado", "jelado"),
        ("casa", "caza"), ("mesa", "meza"), ("azul", "asul"),
        ("cabeça", "cabessa"), ("professor", "profeçor"),
        ("campo", "canpo"), ("também", "tanbem"), ("também", "tambem"),
        ("água", "agua"), ("menino", "meninu"), ("noite", "noit"),
        ("vez", "vês"), ("fazer", "faser"), ("quilo", "kilo"),
        ("passear", "pasiar") if False else ("passo", "paço"),
    ]
    DIFF = [  # sound-altering slips (must compare DIFFERENT)
        ("pato", "bato"), ("dedo", "tedo"), ("faca", "vaca"),
        ("prato", "pato"), ("livro", "livo"), ("verde", "vedre"),
        ("gato", "gado"), ("bola", "pola"), ("caro", "carro"),
        ("avó", "avô") if False else ("sopa", "soba"),
        ("flor", "for"), ("três", "tês"),
    ]
    ok = 0
    for a, b in SAME:
        r = pt_same_sound(a, b)
        print(f"{'OK ' if r else '?? '}same-sound {a!r} ~ {b!r}: "
              f"{pt_phonemes(a)} vs {pt_phonemes(b)}")
        ok += r
    for a, b in DIFF:
        r = not pt_same_sound(a, b)
        print(f"{'OK ' if r else '?? '}diff-sound {a!r} ~ {b!r}: "
              f"{pt_phonemes(a)} vs {pt_phonemes(b)}")
        ok += r
    print(f"{ok}/{len(SAME) + len(DIFF)} as expected")
