"""One-off diagnostic: dump subword-level token/label alignment for a
handful of real training examples. Word-level tags are already confirmed
correct (see synthetic_corruption.json), but the model completely fails to
learn any error class even with focal loss + class weights + many epochs
and real loss reduction -- which points at a possible subword alignment
bug in BIODataset (train_linguistic_model.py) rather than a class-imbalance
problem. This prints exactly what the model actually trains on, subtoken
by subtoken, so we can see with our own eyes whether the B-/I- tags land
on the right pieces.
"""
import json
from transformers import AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-base"
LABEL_LIST = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer class: {type(tokenizer).__name__}  is_fast={tokenizer.is_fast}")

synthetic = json.load(open("../data/linguistic/synthetic_corruption.json"))
# Pick a few examples that actually contain a non-O tag.
examples = [ex for ex in synthetic if any(t != "O" for t in ex["bio_tags"])][:5]

for ex in examples:
    tokens, tags = ex["tokens"], ex["bio_tags"]
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=64)
    word_ids = enc.word_ids()
    subtokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    labels = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)
        elif wid != prev_word_id:
            labels.append(LABEL2ID.get(tags[wid], 0))
        else:
            tag = tags[wid]
            if tag.startswith("B-"):
                tag = "I-" + tag[2:]
            labels.append(LABEL2ID.get(tag, 0))
        prev_word_id = wid

    print("\n=== ", " ".join(tokens), " ===")
    print("word-level tags:", list(zip(tokens, tags)))
    print(f"{'subtoken':<15} {'word_id':>7} {'label_id':>8}  label")
    for st, wid, lid in zip(subtokens, word_ids, labels):
        print(f"{st:<15} {str(wid):>7} {lid:>8}  {ID2LABEL.get(lid, '-100 (ignored)')}")
