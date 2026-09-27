"""
Rule-based synthetic corruption generator for the linguistic error module
(Thesis Ch.3, "Rule-Based Synthetic Data Generation").

Takes a clean sentence and injects exactly one error of a chosen type
(ERR-PHONO, ERR-ORTHO, ERR-SEG), returning word-level BIO labels aligned to
the corrupted sentence's own tokenization. Ground truth is correct *by
construction* (we made the error, so we know exactly where it is), which is
the whole point of the synthetic half of the data strategy: it buys labeled
volume and balanced category coverage that real corpora (BEA-2019 etc.)
can't guarantee for this specific fine-grained taxonomy.

Base clean sentences: originally a 20-sentence hand-written placeholder.
Real training showed this was a real problem, not just a "should improve
later" note -- Stage 1 (synthetic-only) hit eval F1 = 1.0, which turned out
to be memorization (train/eval sentences shared the same 20 templates), not
real generalization. Replaced with a pool of ~600k real, diverse English
sentences (`data/linguistic/clean_base_sentences.json`, filtered from
sentence-transformers/wikipedia-en-sentences on HF -- 7.87M sentences total,
filtered down to 5-18 word, plain-punctuation sentences) via
`load_base_sentences()` below. Falls back to the original small list only
if that file is missing (e.g. a from-scratch smoke test).
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

_BASE_SENTENCES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "linguistic", "clean_base_sentences.json")


def load_base_sentences(path: str = _BASE_SENTENCES_PATH) -> list:
    """Loads the large (~600k) clean base-sentence pool if present, else
    falls back to the small hand-written SEED_SENTENCES placeholder."""
    if os.path.exists(path):
        with open(path) as f:
            sentences = json.load(f)
        print(f"[synthetic_corruption] loaded {len(sentences)} base sentences from {path}")
        return sentences
    print(f"[synthetic_corruption] {path} not found -- falling back to {len(SEED_SENTENCES)} seed sentences")
    return SEED_SENTENCES


# ---------------------------------------------------------------------------
# Fallback seed corpus of clean, grade-appropriate sentences -- only used
# when clean_base_sentences.json (see load_base_sentences above) is absent.
# ---------------------------------------------------------------------------
SEED_SENTENCES = [
    "The dog runs to the park every day.",
    "She likes to read books about animals.",
    "We went to the beach last summer.",
    "My best friend has a big garden.",
    "The teacher gave us homework about plants.",
    "He plays soccer with his brother after school.",
    "The cat sleeps on the warm bed.",
    "They watched a movie together on Friday.",
    "I want to visit the science museum soon.",
    "Our class is learning about the water cycle.",
    "The bird built a nest in the tall tree.",
    "She drew a picture of her family.",
    "The children played games in the yard.",
    "He forgot his lunch box at home today.",
    "The river flows through the small village.",
    "We planted flowers in the school garden.",
    "The boy found a coin under the table.",
    "My mother bakes bread every weekend.",
    "The students wrote a story about space.",
    "The farmer feeds the animals every morning.",
]

VOICING_PAIRS = [("p", "b"), ("t", "d"), ("f", "v"), ("k", "g"), ("s", "z")]
VOICING_MAP = {a: b for a, b in VOICING_PAIRS} | {b: a for a, b in VOICING_PAIRS}

ORTHO_RULES = [
    (re.compile(r"ph"), "f"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"ss"), "s"),
    (re.compile(r"ie"), "ei"),
    (re.compile(r"tion"), "shun"),
]

HOMOPHONES = {
    "there": "their", "their": "there", "to": "too", "too": "to",
    "your": "you're", "you're": "your", "its": "it's", "it's": "its",
    "here": "hear", "hear": "here", "right": "write", "write": "right",
    "week": "weak", "weak": "week", "sun": "son", "son": "sun",
}

LABELS = ["ERR-PHONO", "ERR-ORTHO", "ERR-SEG"]


@dataclass
class CorruptionResult:
    original: str
    corrupted: str
    tokens: list       # corrupted sentence, whitespace-tokenized
    bio_tags: list      # same length as tokens
    label: str
    note: str


def _tag_single_token(tokens: list, idx: int, label: str) -> list:
    tags = ["O"] * len(tokens)
    tags[idx] = f"B-{label}"
    return tags


def corrupt_phonological(sentence: str, rng: random.Random) -> Optional[CorruptionResult]:
    words = sentence.split()
    candidates = [i for i, w in enumerate(words) if any(c in VOICING_MAP for c in w.lower())]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    word = words[idx]
    chars = list(word)
    sub_positions = [i for i, c in enumerate(chars) if c.lower() in VOICING_MAP]
    pos = rng.choice(sub_positions)
    orig_char = chars[pos]
    repl = VOICING_MAP[orig_char.lower()]
    if orig_char.isupper():
        repl = repl.upper()
    chars[pos] = repl
    new_word = "".join(chars)
    if new_word == word:
        return None
    new_words = words.copy()
    new_words[idx] = new_word
    tags = _tag_single_token(new_words, idx, "PHONO")
    return CorruptionResult(sentence, " ".join(new_words), new_words, tags, "ERR-PHONO",
                             f"voicing substitution '{orig_char}'->'{repl}' in {word!r}->{new_word!r}")


def corrupt_orthographic(sentence: str, rng: random.Random) -> Optional[CorruptionResult]:
    words = sentence.split()
    # Try homophone swap first (50% of the time), else contextual rule violation.
    homophone_candidates = [i for i, w in enumerate(words)
                             if re.sub(r"[^a-zA-Z']", "", w).lower() in HOMOPHONES]
    rule_candidates = []
    for i, w in enumerate(words):
        for pat, _ in ORTHO_RULES:
            if pat.search(w.lower()):
                rule_candidates.append(i)
                break

    use_homophone = homophone_candidates and (not rule_candidates or rng.random() < 0.5)
    if use_homophone:
        idx = rng.choice(homophone_candidates)
        word = words[idx]
        core = re.sub(r"[^a-zA-Z']", "", word).lower()
        repl = HOMOPHONES[core]
        # preserve trailing punctuation
        suffix = word[len(core):] if word.lower().startswith(core) else ""
        new_word = repl + suffix
        new_words = words.copy()
        new_words[idx] = new_word
        tags = _tag_single_token(new_words, idx, "ORTHO")
        return CorruptionResult(sentence, " ".join(new_words), new_words, tags, "ERR-ORTHO",
                                 f"homophone swap {word!r}->{new_word!r}")
    elif rule_candidates:
        idx = rng.choice(rule_candidates)
        word = words[idx]
        new_word = word
        for pat, repl in ORTHO_RULES:
            if pat.search(word.lower()):
                new_word = pat.sub(repl, word, count=1)
                break
        if new_word == word:
            return None
        new_words = words.copy()
        new_words[idx] = new_word
        tags = _tag_single_token(new_words, idx, "ORTHO")
        return CorruptionResult(sentence, " ".join(new_words), new_words, tags, "ERR-ORTHO",
                                 f"contextual rule violation {word!r}->{new_word!r}")
    return None


def corrupt_segmentation(sentence: str, rng: random.Random) -> Optional[CorruptionResult]:
    words = sentence.split()
    if len(words) < 2:
        return None
    # 50/50 between hyposegmentation (merge two words) and
    # hypersegmentation (split one word into two).
    if rng.random() < 0.5:
        # merge: pick adjacent pair, fuse into one token
        idx = rng.randrange(len(words) - 1)
        merged = words[idx] + words[idx + 1]
        new_words = words[:idx] + [merged] + words[idx + 2:]
        tags = _tag_single_token(new_words, idx, "SEG")
        return CorruptionResult(sentence, " ".join(new_words), new_words, tags, "ERR-SEG",
                                 f"hyposegmentation: merged {words[idx]!r}+{words[idx+1]!r} -> {merged!r}")
    else:
        # split: pick a word with length >= 4, insert a space inside it
        candidates = [i for i, w in enumerate(words) if len(re.sub(r'[^a-zA-Z]', '', w)) >= 4]
        if not candidates:
            return None
        idx = rng.choice(candidates)
        word = words[idx]
        core_len = len(re.sub(r'[^a-zA-Z]', '', word))
        split_at = rng.randint(2, core_len - 2) if core_len > 4 else core_len // 2
        part1, part2 = word[:split_at], word[split_at:]
        new_words = words[:idx] + [part1, part2] + words[idx + 1:]
        tags = ["O"] * len(new_words)
        tags[idx] = "B-SEG"
        tags[idx + 1] = "I-SEG"
        return CorruptionResult(sentence, " ".join(new_words), new_words, tags, "ERR-SEG",
                                 f"hypersegmentation: split {word!r} -> {part1!r}+{part2!r}")


CORRUPTORS = {
    "ERR-PHONO": corrupt_phonological,
    "ERR-ORTHO": corrupt_orthographic,
    "ERR-SEG": corrupt_segmentation,
}


def generate_corpus(n: int, base_sentences: Optional[list] = None,
                     rng: Optional[random.Random] = None) -> list:
    rng = rng or random.Random(0)
    base_sentences = base_sentences if base_sentences is not None else load_base_sentences()
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        label = rng.choice(LABELS)
        sentence = rng.choice(base_sentences)
        result = CORRUPTORS[label](sentence, rng)
        if result is not None:
            out.append(result)
    return out
