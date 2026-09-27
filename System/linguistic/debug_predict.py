"""One-off diagnostic: run the trained final_model directly (no Trainer, no
compute_metrics) on a handful of training examples it has seen many times,
and print the FULL per-token probability distribution, not just argmax.

Point: eval_precision/recall/f1 have been stuck at exactly 0.0 through
several loss-function changes (plain CE, weighted CE, focal loss), even
though train/eval loss genuinely goes down and subword label alignment is
confirmed correct (see debug_alignment.py output). This checks whether the
model has learned *any* signal at all for the rare error classes (e.g.
15-20% probability on B-PHONO where gold is B-PHONO, just not enough to
win the argmax) versus literally zero signal (near-uniform/all-mass-on-O
regardless of input), which points at two very different next fixes.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "./final_model"
LABEL_LIST = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"device: {device}")

synthetic = json.load(open("../data/linguistic/synthetic_corruption.json"))
examples = [ex for ex in synthetic if any(t != "O" for t in ex["bio_tags"])][:8]

for ex in examples:
    tokens, tags = ex["tokens"], ex["bio_tags"]
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True,
                     max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1).tolist()
    word_ids = enc.word_ids()
    subtoks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    print("\n===", " ".join(tokens), "===")
    for st, wid, p, pr in zip(subtoks, word_ids, preds, probs):
        gold = tags[wid] if wid is not None else "-"
        top3 = sorted(zip(LABEL_LIST, pr.tolist()), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{l}:{v:.3f}" for l, v in top3)
        flag = "  <-- GOLD IS NON-O" if gold != "O" and gold != "-" else ""
        print(f"{st:<12} gold={gold:<10} pred={ID2LABEL[p]:<10} top3=[{top3_str}]{flag}")
