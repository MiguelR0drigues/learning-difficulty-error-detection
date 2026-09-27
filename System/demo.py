"""End-to-end interactive demo of the thesis system.

Run from the System/ directory:
    python demo.py                # interactive session
    python demo.py --examples     # scripted walkthrough with sample inputs

Commands in interactive mode:
    text <sentence>                e.g.  text I go to scool some times
    sub <a> <b> <answer>           e.g.  sub 52 17 45      (52-17, student wrote 45)
    frac <n1/d1> <n2/d2> <ans>     e.g.  frac 1/2 1/3 2/5  (1/2+1/3, student wrote 2/5)
    profile                        show the accumulated difficulty profile
    quit

Pipeline demonstrated: each command routes the response to the right
detection module (Ch.4); every detected error label is accumulated into a
session profile, exactly the aggregation the profiling module clusters on.
"""
from __future__ import annotations

import sys
from collections import Counter
from fractions import Fraction

sys.path.insert(0, "math")
sys.path.insert(0, "linguistic")

from mal_rules import detect_subtraction_error, detect_fraction_add_error  # noqa: E402

LABEL_LIST = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

_MODEL = None
_TOKENIZER = None
_DEVICE = "cpu"


def load_linguistic_model(model_dir="linguistic/final_model"):
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return True
    import os
    if not os.path.isdir(model_dir):
        print(f"[!] {model_dir} not found -- train first (see RUN_ON_GPU.md)")
        return False
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    print("Loading linguistic model (first call only)...")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    _MODEL = AutoModelForTokenClassification.from_pretrained(model_dir, torch_dtype=torch.float32)
    _MODEL.eval()
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL.to(_DEVICE)
    print(f"Loaded on {_DEVICE}.")
    return True


def analyze_text(sentence: str) -> list:
    """Returns list of (span_text, error_type) detected in the sentence."""
    import torch
    if not load_linguistic_model():
        return []
    tokens = sentence.split()
    enc = _TOKENIZER(tokens, is_split_into_words=True, truncation=True,
                     max_length=64, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        preds = _MODEL(**enc).logits.argmax(dim=-1)[0].tolist()
    word_ids = enc.word_ids()
    # first-subtoken prediction per word
    word_pred = {}
    for wid, p in zip(word_ids, preds):
        if wid is not None and wid not in word_pred:
            word_pred[wid] = ID2LABEL[p]
    spans, cur = [], None
    for i, tok in enumerate(tokens):
        tag = word_pred.get(i, "O")
        if tag.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = [tok, tag[2:]]
        elif tag.startswith("I-") and cur and cur[1] == tag[2:]:
            cur[0] += " " + tok
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return [(s[0], s[1]) for s in spans]


class SessionProfile:
    def __init__(self):
        self.counts = Counter()
        self.n_responses = 0
        self.n_errors = 0

    def add(self, labels: list):
        self.n_responses += 1
        for lab in labels:
            if lab not in ("CORRECT", "OTHER", "UNKNOWN"):
                self.counts[lab] += 1
                self.n_errors += 1

    def show(self):
        print(f"\n--- Session profile ({self.n_responses} responses, {self.n_errors} in-scope errors) ---")
        if not self.counts:
            print("No in-scope errors detected yet.")
            return
        total = sum(self.counts.values())
        for lab, c in self.counts.most_common():
            bar = "#" * int(30 * c / total)
            print(f"  {lab:<14} {c:>3}  ({c/total:5.1%}) {bar}")
        dom = self.counts.most_common(1)[0][0]
        print(f"Dominant error type so far: {dom}")
        print("(In deployment, this vector -- proportions + error rate + trend --")
        print(" is what the profiling module clusters across students.)")


def handle(cmd: str, profile: SessionProfile):
    parts = cmd.strip().split()
    if not parts:
        return
    op = parts[0].lower()
    if op == "text":
        sentence = cmd.strip()[5:]
        spans = analyze_text(sentence)
        if spans:
            for text, etype in spans:
                print(f"  [{etype}] \"{text}\"")
            profile.add([etype for _, etype in spans])
        else:
            print("  no in-scope errors detected")
            profile.add([])
    elif op == "sub" and len(parts) == 4:
        a, b, ans = int(parts[1]), int(parts[2]), int(parts[3])
        label = detect_subtraction_error(a, b, ans)
        print(f"  {a} - {b} = {ans}  ->  {label}")
        profile.add([label])
    elif op == "frac" and len(parts) == 4:
        f1, f2 = Fraction(parts[1]), Fraction(parts[2])
        ans = Fraction(parts[3])
        label = detect_fraction_add_error(f1.numerator, f1.denominator,
                                          f2.numerator, f2.denominator, ans)
        print(f"  {parts[1]} + {parts[2]} = {parts[3]}  ->  {label}")
        profile.add([label])
    elif op == "profile":
        profile.show()
    elif op in ("quit", "exit", "q"):
        raise SystemExit
    else:
        print("  commands: text <sentence> | sub <a> <b> <ans> | frac <a/b> <c/d> <ans> | profile | quit")


EXAMPLES = [
    "text I go to scool some times with my sister",
    "text My favourite subject is siance becaus it is fun",
    "sub 52 17 45",
    "sub 305 127 288",
    "sub 52 17 35",
    "frac 1/2 1/3 2/5",
    "frac 1/2 1/3 5/6",
    "profile",
]

if __name__ == "__main__":
    profile = SessionProfile()
    if "--examples" in sys.argv:
        for cmd in EXAMPLES:
            print(f"\n>>> {cmd}")
            handle(cmd, profile)
    else:
        print(__doc__)
        while True:
            try:
                handle(input("\n> "), profile)
            except (EOFError, KeyboardInterrupt):
                break
            except SystemExit:
                break
            except Exception as e:
                print(f"  error: {e}")
