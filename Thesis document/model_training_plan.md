# Model & Training Plan (Next Steps)

Builds on `data_strategy_plan.md`. Important framing before the steps: **only the linguistic module needs a pretrained "base model"** in the deep-learning sense. The math module is symbolic/rule-based (per Ch.3's own design), and profiling is unsupervised clustering — neither has a "base model" to pick.

---

## 1. Linguistic Error Detection Module

### 1.1 Data acquisition
- `datasets.load_dataset("bea2019st/wi_locness")` — real training/dev data.
- `datasets.load_dataset("jhu-clsp/jfleg")` — held-out fluency eval set.
- Inspect actual record schema before writing any loader (source sentence, corrected sentence, edit spans).

### 1.2 Edit extraction & taxonomy mapping
- Install ERRANT (`pip install errant`) — takes parallel original/corrected sentences and outputs classified edits in M2 format (Missing/Replacement/Unnecessary + type tags like SPELL, ORTH, PUNCT, etc.).
- Map ERRANT's generic types onto the thesis's ERR-PHONO / ERR-ORTHO / ERR-SEG taxonomy with a heuristic layer:
  - `SPELL`-type edits → phonetic distance (soundex/metaphone) between student token and gold token: high phonetic similarity ⇒ ERR-PHONO (sound-based confusion), low similarity but plausible visual/rule confusion ⇒ ERR-ORTHO.
  - Whitespace insert/delete edits → ERR-SEG (hypo/hyper-segmentation).
  - Everything else (verb tense, punctuation, word order, etc.) is out of scope — discard or keep as an `OTHER` bucket for transparency, not used in training.
- This mapping is a genuine methodological step worth documenting explicitly in the (eventual) Ch.3/Ch.4 rewrite — it's not in the literature ready-made for this exact taxonomy.

### 1.3 Synthetic data generation
- Base clean sentences: the "correct" side of BEA-2019/JFLEG, or a general corpus (Wikipedia simple-English dumps), filtered for grade-appropriate vocabulary/length.
- Corruption rules (already sketched in Ch.3): probabilistic phoneme-pair substitution (p/b, t/d, f/v, k/g), grapheme-rule violations (contextual spelling confusions), word fusion/splitting.
- Output: (corrupted_sentence, BIO_labels) pairs at controllable volume — this is what balances the categories real data under-represents.

### 1.4 Label alignment
- Tokenize with the chosen model's tokenizer; align word-level BIO labels to subword tokens via `tokenizer(...).word_ids()` (standard HF token-classification recipe — first subtoken gets the label, rest get `-100`/ignored, or repeat with I- tag).

### 1.5 Base model choice: **DeBERTa-v3-base** (`microsoft/deberta-v3-base`)
- Current evidence (checked directly, not from stale priors): for token-classification/NER-style precision, DeBERTa-v3 still outperforms newer encoders like ModernBERT, which optimizes for speed/long-context rather than label precision — the opposite trade-off from what this task needs. If compute is tight, drop to `deberta-v3-small` before considering a different architecture.
- Use `AutoModelForTokenClassification` with a `num_labels` head matching the BIO tag set.

### 1.6 Training curriculum (adapted from GECToR's staged approach, which fits H2 naturally)
1. **Stage 1** — pretrain the classification head (and lightly fine-tune the encoder) on large-volume synthetic data only.
2. **Stage 2** — fine-tune on real BEA-2019 data (taxonomy-mapped).
3. **Stage 3** — fine-tune on a balanced mix of synthetic + real.
- Keep checkpoints from each stage — this directly produces the synthetic-only vs. real-only vs. combined comparison Ch.3/Ch.5 already call for for H2.

### 1.7 Evaluation
- `seqeval` for span-level precision/recall/F1 per error type (PHONO/ORTHO/SEG).
- Confusion matrix across categories.
- JFLEG held out strictly for final generalization check, never trained on.

---

## 2. Mathematical Error Detection Module

### 2.1 Data
- DeepMind `mathematics_dataset` (`pip install mathematics_dataset`) — generate correct arithmetic problems (focus on the `arithmetic` module for subtraction, and fraction-related generation for the fraction misconceptions).
- MaE dataset (HF `nanote/algebra_misconceptions`) — real, expert-validated, used for validation/seed, not primary training volume.

### 2.2 Malrule simulator (this is the "model" here — deterministic, not learned)
- For each correct problem, apply each malrule transformation (SFL, borrow-from-zero, borrow-no-decrement, frac-add, frac-compare) programmatically (SymPy for symbolic correctness of the "correct" path) and record (problem, malrule_answer, label).
- Detection at inference time = reverse simulation: given a problem and a student's (wrong) answer, run all malrule simulations and check which one reproduces the student's answer exactly → that's the predicted label. No answer matches → `OTHER_ERROR`.

### 2.3 Optional light classifier for ambiguous cases
- When multiple malrules coincide on the same wrong answer, or none match cleanly (e.g., due to noise), a small supervised model (gradient boosting or logistic regression on hand-built features: digit-wise differences, whether borrowing was needed, etc.) trained on the synthetic malrule-labeled set can arbitrate. This is a minor addition, not the core method.

### 2.4 Evaluation
- Accuracy/F1 per bug type, confusion matrix.
- Cross-check against MaE's real algebra misconceptions (even though MaE is algebra, not pure arithmetic — it validates whether the malrule-matching approach generalizes beyond the hand-authored bug list).

---

## 3. Student Profiling Module

### 3.1 No independent data — consumes outputs of modules 1–2
- Per-response error-type labels/counts, aggregated per synthetic "student" across simulated sessions/time steps.

### 3.2 Simulation harness
- Define profile archetypes (e.g., phonological-dominant, segmentation-dominant, fraction-misconception-dominant, mixed/low-difficulty).
- Sample multiple responses per synthetic student biased toward their archetype's error distribution, across several simulated time points.

### 3.3 Feature vector per student
- Per-error-type rate/frequency, trend over time (e.g., slope of error rate across sessions).

### 3.4 Clustering
- K-Means (elbow/silhouette to pick k) and/or HDBSCAN (no fixed k, handles noise/outlier students) — try both, compare.

### 3.5 Evaluation
- Silhouette score, Davies-Bouldin index.
- Since ground-truth archetypes are known (synthetic), Adjusted Rand Index against the true archetype labels — a strong, honest validation of H3 that most real-data studies can't do.

---

## 4. Recommended Execution Order

1. Environment setup: `venv`, `transformers`, `datasets`, `errant`, `sympy`, `scikit-learn`, `mathematics_dataset`.
2. Download and manually inspect BEA-2019, JFLEG, MaE, and sample `mathematics_dataset` output — confirm real schemas before coding loaders.
3. Build the ERRANT extraction + taxonomy-mapping pipeline (linguistic side — most complex, do it first).
4. Build the linguistic synthetic corruption generator.
5. Train DeBERTa-v3-base in the three stages (synthetic-only / real-only / combined) — this alone answers a first pass at H1 and H2.
6. In parallel or after: build the math malrule simulator (fast, no GPU needed, good for early results while the linguistic model trains).
7. Only once 1–2 produce real outputs: build the profiling simulation harness and clustering.
