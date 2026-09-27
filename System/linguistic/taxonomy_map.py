"""
Maps ERRANT's generic GEC error types onto the thesis's fine-grained
taxonomy: ERR-PHONO / ERR-ORTHO / ERR-SEG (Ch.3, Section "Label Scheme").

Revision note (validated against real BEA-2019 gold edits, not just toy
examples): an earlier version used a blanket "same Metaphone code -> treat
as homophone confusion" rule for ANY single-token replacement, regardless of
ERRANT's own type. Running it over ~65k real BEA-2019 edits showed this is
too permissive -- Metaphone is coarse enough that unrelated content-word
substitutions ERRANT itself tags as word-choice errors (R:NOUN, R:VERB,
etc., e.g. 'groceries'->'grocers', 'pick'->'peak') collide by coincidence
and got mislabeled as ERR-ORTHO. Real homophone confusion (their/there,
to/too, its/it's...) is a closed-class phenomenon, so it is now matched
against a curated lexicon instead of a generic phonetic-similarity fallback.
Similarly, a pure-casing edit ('Journalist'->'journalist', no space change
at all) was being caught by an overly loose "normalized forms match"
whitespace check; that check now requires an actual difference in the
number of space characters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import jellyfish

THESIS_LABELS = ["ERR-PHONO", "ERR-ORTHO", "ERR-SEG", "OTHER"]

# Curated closed-class homophone pairs (extendable). Deliberately NOT a
# generic phonetic-similarity check -- see module docstring for why.
HOMOPHONE_LEXICON = {
    "there": "their", "their": "there", "theyre": "there",
    "to": "too", "too": "to", "two": "to",
    "your": "youre", "youre": "your",
    "its": "its", "it's": "its",
    "here": "hear", "hear": "here",
    "right": "rite", "rite": "right", "write": "right",
    "week": "weak", "weak": "week",
    "sun": "son", "son": "sun",
    "then": "than", "than": "then",
    "know": "no", "no": "know",
    "new": "knew", "knew": "new",
    "one": "won", "won": "one",
    "wear": "where", "where": "wear",
    "peace": "piece", "piece": "peace",
    "whole": "hole", "hole": "whole",
}


def _normalize(tok: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", tok).lower()


def _strip_edge_nonalpha(s: str) -> str:
    """Strip leading/trailing runs of non-letter characters (punctuation,
    and any spaces adjacent to them). Used so that shared sentence-edge
    punctuation (e.g. a trailing period ERRANT includes in the edit span)
    doesn't get confused with genuine internal word-boundary punctuation."""
    s = re.sub(r"^[^a-zA-Z]+", "", s)
    s = re.sub(r"[^a-zA-Z]+$", "", s)
    return s


def _is_segmentation_diff(o_str: str, c_str: str) -> bool:
    """True only when this is a genuine word-boundary change: after
    stripping shared edge punctuation, the same letters remain but grouped
    into a different number of space-separated tokens, with no punctuation
    left INSIDE either string.

    Third real-data finding (BEA-2019): edits like 'we' -> '. We' or
    ', however' -> '. However ,' look like segmentation changes under a
    naive space-count check, but they are sentence-boundary / run-on-
    sentence fixes (a period was inserted at the edge) -- not a word being
    merged or split. Stripping edge punctuation before comparing correctly
    routes these to OTHER instead of ERR-SEG, while still catching genuine
    cases like 'everyday.' -> 'every day.' (trailing period is shared and
    irrelevant; the real change is the internal space)."""
    o2, c2 = _strip_edge_nonalpha(o_str), _strip_edge_nonalpha(c_str)
    if re.search(r"[^\w\s]", o2) or re.search(r"[^\w\s]", c2):
        return False  # punctuation remains inside -> not pure segmentation
    no, nc = _normalize(o2), _normalize(c2)
    if no != nc or not no:
        return False
    return o2.count(" ") != c2.count(" ")


@dataclass
class MappedEdit:
    o_str: str
    c_str: str
    errant_type: str
    thesis_label: str
    reason: str


def map_edit(o_str: str, c_str: str, errant_type: str) -> MappedEdit:
    if _is_segmentation_diff(o_str, c_str):
        return MappedEdit(o_str, c_str, errant_type, "ERR-SEG", "whitespace/token-count difference, same letters")

    if errant_type == "R:ORTH":
        # ERRANT's own "ORTH" bucket covers case AND whitespace edits.
        # Whitespace already handled above; remaining ORTH edits here are
        # pure capitalization, out of this thesis's scope.
        return MappedEdit(o_str, c_str, errant_type, "OTHER", "capitalization only")

    if errant_type == "R:SPELL":
        no, nc = _normalize(o_str), _normalize(c_str)
        if not no or not nc:
            return MappedEdit(o_str, c_str, errant_type, "OTHER", "empty token after normalization")
        same_sound = jellyfish.metaphone(no) == jellyfish.metaphone(nc)
        if same_sound:
            return MappedEdit(o_str, c_str, errant_type, "ERR-PHONO",
                               "same Metaphone code: phonetically-faithful mis-spelling")
        return MappedEdit(o_str, c_str, errant_type, "ERR-ORTHO",
                           "different Metaphone code: visual/rule-memory slip")

    # Homophone confusions between two REAL, correctly-spelled words
    # (their/there, to/too, its/it's...) are the "Homophone confusions"
    # sub-category the thesis (Ch.3) explicitly lists under ERR-ORTHO.
    # ERRANT does not tag these as R:SPELL, so we recover them via a
    # curated closed-class lexicon (NOT a generic phonetic check -- see
    # module docstring).
    no, nc = _normalize(o_str), _normalize(c_str)
    if no in HOMOPHONE_LEXICON and HOMOPHONE_LEXICON[no] == nc:
        return MappedEdit(o_str, c_str, errant_type, "ERR-ORTHO",
                           f"homophone confusion (lexicon match); ERRANT tagged it as {errant_type}")

    return MappedEdit(o_str, c_str, errant_type, "OTHER", f"out of scope (errant type {errant_type})")


def extract_and_map(annotator, orig_sentence: str, corrected_sentence: str) -> list:
    orig = annotator.parse(orig_sentence)
    cor = annotator.parse(corrected_sentence)
    edits = annotator.annotate(orig, cor)
    return [map_edit(e.o_str, e.c_str, e.type) for e in edits]
