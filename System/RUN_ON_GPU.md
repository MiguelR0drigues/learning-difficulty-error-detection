# Running real training on your RTX 5070 (12GB)

This sandbox has no GPU, so everything here was only smoke-tested on CPU
with a tiny model/data slice. Do the real runs on your own machine.

## 1. Linguistic model (deep learning, the main AI/ML deliverable)

**Important: use Python 3.11 or 3.12 for this venv, NOT 3.14.** As of
mid-2026, spaCy's dependencies (thinc, blis) still don't have reliable
prebuilt wheels for Python 3.14, so pip tries to compile them from source
and fails (Cython/numpy ABI errors). Python 3.11/3.12 have full prebuilt
wheel support for the entire stack (torch, transformers, spacy) and are
what this System was actually tested with (Python 3.10 in the sandbox).
This doesn't require removing 3.14 -- just target a different interpreter
when creating the venv.

**Also important: your RTX 5070 needs CUDA 12.8+ wheels (`cu128`), not
`cu121`.** The RTX 50-series (Blackwell, compute capability sm_120) is
too new for CUDA 12.1 kernels -- installing the `cu121` build gives you
*a* working torch, but it silently can't use your GPU (or errors out).

```bash
# 1. Install Python 3.11 (or 3.12) from python.org if you don't have it
#    already -- this does not replace your existing Python 3.14.

# 2. Create the venv with that specific interpreter (Windows py launcher):
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Install everything EXCEPT torch from requirements.txt first:
pip install -r ../requirements.txt --no-deps  # or just skip the torch line manually

# 4. Install the correct CUDA build of torch for a Blackwell/RTX 50-series GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 5. Verify the GPU is actually detected before doing anything else:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Must print: True  NVIDIA GeForce RTX 5070
# If it still prints False, check https://pytorch.org/get-started/locally/
# for the current recommended index (cu128 may have been superseded by a
# newer one like cu129/cu130 by the time you read this -- use whatever
# that page recommends for your CUDA driver version).

# 6. Now the spaCy model should install cleanly (prebuilt wheels for 3.11/3.12):
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# 7. Real training run (deberta-v3-base, full datasets, 3-stage curriculum):
set SMOKE_TEST=0
python train_linguistic_model.py
# (PowerShell: $env:SMOKE_TEST="0")
```

Expect this to take somewhere from tens of minutes to a couple hours
depending on epoch counts — 12GB VRAM comfortably fits deberta-v3-base at
batch size 16-32 with sequence length 64. Watch `nvidia-smi` if you want to
push batch size higher.

**What to look at when it's done:** compare the eval F1 (per Stage 1 vs 2 vs
3, printed by the script and via `compute_metrics`) — that comparison is
the empirical answer to H2 (does combining synthetic + real data beat
either alone?). Also compare against BEA-2019's own published F0.5
baselines (Bryant et al. 2019) for a sanity-check reference point, though
note our label taxonomy (PHONO/ORTHO/SEG) isn't the same as ERRANT's full
error-type set, so it's not a strict apples-to-apples comparison.

**Done since first draft of this file (2026-07-04):**
- Synthetic seed sentences: the 20-template placeholder was replaced with a
  ~600k-sentence Wikipedia pool; `synthetic_corruption.json` is now 1M examples.
- Real data expanded 2.7x: `data/linguistic/real_bio_combined.json` (6,915
  sentences) = BEA-2019 W&I+LOCNESS (2,584) + **FCE v2.1 train+dev (4,331)**,
  both built with the same m2_parser -> taxonomy_map -> BIO pipeline. The
  training script picks this file up automatically (falls back to the
  BEA-only file if missing). FCE *test* is kept out of the training pool as
  an optional held-out benchmark: `data/linguistic/fce_test_bio.json` (383
  sentences). FCE licence: non-commercial research, cite Yannakoudakis et
  al. (2011) in the thesis.
- Stage 2 epochs reduced 15 -> 8 to match the 3x larger real corpus.

**Reading smoke-test output:** `SMOKE_TEST=1` (the default!) trains on 16
sentences and evaluates on 6 — its metrics are meaningless by design and the
summary now prints a banner saying so. Only `SMOKE_TEST=0` output is a real
result. (This caused a false alarm on 2026-07-04: a smoke run's F1=0.05 was
read as a training failure.)

## 1b. Scientific-rigor runs (added 2026-07-05)

Three GPU jobs, in priority order. All from `System/linguistic/`, venv active.

**(A) Multi-seed + paired H2 CI — the important one (~4h/seed):**
```bash
# seed 1 (repeat with SEED=2, SEED=3 if time allows; seed 0 = existing run)
set SMOKE_TEST=0
set SEED=1
set SAVE_STAGE2=1
python train_linguistic_model.py
python eval_h2_ci.py        # paired bootstrap CI of (final - stage2) on real_eval
```
Each seed produces `h2_paired_ci_seed<S>.json`. With 3 seeds: report mean±sd
of the H2 difference + per-seed CIs in Ch.5. This turns +4.2 from an
observation into a demonstrated effect.

**(B) BERT-base baseline (one run, ~2-3h):** edit MODEL_NAME to
"bert-base-cased" (or set via env if you add one), run SMOKE_TEST=0 SEED=0,
report final-on-real_eval vs DeBERTa's. Justifies the encoder choice.

**(C) Synthetic-scale ablation (3 runs, ~30-90min each):** temporarily point
stage 1/3 at subsets (10k / 100k / 1M) of synthetic_corruption.json (e.g.
slice after loading) and record final-on-real_eval F1 per size → "F1 vs
synthetic scale" curve for Ch.5. More data prep than GPU time.

## 2. Math ML classifier (already run once, in this sandbox, on CPU)

No GPU needed — `math/train_ml_classifier.py` already ran to completion
here using `GradientBoostingClassifier` (scikit-learn). Results are in
`data/math_ml_vs_symbolic_results.json`. You can re-run it yourself, or
extend it (try other classifiers, add the same clean-vs-noisy comparison
for the fraction malrules, tune hyperparameters) — it finishes in seconds.

## 3. Profiling

Also CPU-only, already validated (`data/profiling_smoke_results.json`).
Nothing to run on GPU here.
