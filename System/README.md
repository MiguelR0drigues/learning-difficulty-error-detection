# System — real thesis implementation (supersedes the POC in `Experimentation/`)

Built following `Thesis document/data_strategy_plan.md` and
`Thesis document/model_training_plan.md`. Scope was narrowed on 2026-07-02
to make sure the thesis has genuine AI/ML content (a purely symbolic
rule-matcher doesn't count for an AI Engineering thesis) while keeping both
the linguistic and math domains. See `RUN_ON_GPU.md` for how to run real
training on Miguel's RTX 5070 (this sandbox has no GPU).

## Status summary

| Module | Code | Tests | Real data | Trained/validated |
|---|---|---|---|---|
| Math — symbolic malrules | done | passing | MaE dataset | deterministic, works today |
| Math — ML classifier | done | passing | validated vs symbolic | **trained + evaluated (CPU, in this sandbox)** |
| Linguistic (error detection) | done | passing | BEA-2019 + JFLEG, real BIO corpus | smoke-tested only — needs GPU run (see `RUN_ON_GPU.md`) |
| Profiling (clustering) | done | validated via cluster-recovery (ARI) | N/A by design | validated (CPU) |

## Math module (`math/`)

**Two methods, deliberately compared** — this directly answers the
thesis's own Q3 ("How effectively do symbolic methods AND simple ML
classifiers identify characteristic arithmetic procedural bugs...").

1. `mal_rules.py` — symbolic malrule simulator (SFL, borrow-from-zero,
   borrow-no-decrement, frac-add, frac-compare), deterministic, no training.
2. `features.py` + `train_ml_classifier.py` — a "simple ML classifier"
   (GradientBoostingClassifier) trained on hand-built numeric features of
   the same problems, compared head-to-head against the symbolic method.

**Real result (already run, `data/math_ml_vs_symbolic_results.json`):**
on exact malrule answers both methods hit 100% accuracy (expected — that's
what both were built for). But on a **noisy** test set (malrule answer
perturbed by ±1 on one digit, simulating a student who applies a buggy
procedure AND makes an unrelated slip), the ML classifier retains 61.7%
accuracy while the symbolic method drops to **0%** — it requires an exact
answer match by construction, so any deviation makes it fail. This is a
clean, real, citable trade-off: symbolic methods are perfectly precise on
exact patterns but brittle to any variation; a classifier trained on the
same generated data generalizes better at some precision cost. Good
material for Ch.5 (results) and Ch.3 (method comparison design).

Real data: `data/math/mae_dataset_raw.json` (55 middle-school algebra
misconceptions, 220 expert-validated examples, from
`github.com/nancyotero-projects/math-misconceptions` — the HF mirror
`nanote/algebra_misconceptions` only hosts loose files, not the structured
JSON, use the GitHub repo instead).

**Not yet done:** the noisy-vs-clean comparison was only run for
subtraction, not fractions — natural extension, and running the ML
classifier against real MaE examples directly (currently only validated
against our own synthetic test split).

## Linguistic module (`linguistic/`)

Deep learning token classification (DeBERTa-v3), 3-stage GECToR-style
curriculum (synthetic → real → combined). **This is the main deep-learning
deliverable and needs a GPU run — see `RUN_ON_GPU.md`.**

- `taxonomy_map.py` — ERRANT → thesis taxonomy (ERR-PHONO/ORTHO/SEG)
  mapper, validated against real BEA-2019 gold edits; caught and fixed 3
  real precision bugs along the way (see docstring).
- `synthetic_corruption.py` — rule-based corruption generator; currently
  draws from a small hand-written seed corpus (20 sentences) — swap for
  real BEA-2019/JFLEG sentences before the real GPU run for better realism.
- `m2_parser.py` / `build_real_bio_corpus.py` — turned real BEA-2019 data
  into **2584 real labeled sentences** (`data/linguistic/bea2019_bio_real.json`).
- `train_linguistic_model.py` — set `SMOKE_TEST=0` for the real run
  (`deberta-v3-base`, full data, 3-4 epochs per stage, batch size 16,
  sized for 12GB VRAM). Default (`SMOKE_TEST=1`) is what was already run
  and validated in this sandbox on CPU with `deberta-v3-xsmall` and a tiny
  data slice — proves the pipeline works, not real performance (eval F1
  was 0 on the smoke test, which is expected given the tiny scale, not a
  negative result).

Real data downloaded from:
- BEA-2019: `https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz`
  (the HF repo `bea2019st/wi_locness` only hosts a loader *script*, which
  current `datasets` versions refuse to run — fetch the tar.gz directly).
- JFLEG: `github.com/keisks/jfleg`.

## Profiling module (`profiling/`)

`simulate_and_cluster.py` — no public dataset exists for this (see
data_strategy_plan.md), so it's a simulation harness: 6 profile
archetypes, simulated multi-session synthetic students, clustering
(KMeans/HDBSCAN) checked against known ground truth via Adjusted Rand
Index.

**Real finding:** recoverability is sensitive to archetype separation
strength and observations-per-student. Weak params: ARI≈0.53-0.58, HDBSCAN
found no clusters. Strong params (more sessions/responses, clearer
archetype signal): KMeans ARI≈0.97, HDBSCAN≈0.90. Practical implication
worth quantifying properly in Ch.5: a real deployment needs a minimum
amount of observed student data before profiles become trustworthy.

## Setup

```
pip install -r requirements.txt --break-system-packages
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl --break-system-packages
```

For GPU training, see `RUN_ON_GPU.md`.

## Data & profiling update (2026-07-04)

Real-data grounding added across modules (all reproducible from scripts):

- **FCE v2.1** (direct download, non-commercial licence, cite Yannakoudakis
  et al. 2011): +4,331 real sentences -> combined real training pool
  `data/linguistic/real_bio_combined.json` (6,915 sentences, 2.7x BEA-only).
  FCE test kept out as held-out benchmark (`fce_test_bio.json`).
- **Holbrook corpus** ('English for the Rejected', 1964; via Roger Mitton):
  real writings of 19 struggling secondary-school children. This is the
  closest public English data to the thesis's target population (literacy
  difficulties, NOT adult L2). Used two ways, never for training:
  1. `data/linguistic/holbrook_bio.json` (531 sentences) — domain-shift
     evaluation set for the linguistic model (`linguistic/holbrook_to_bio.py`).
  2. `data/profiling/holbrook_student_profiles.json` — real per-student
     error distributions. Clustering them (`profiling/holbrook_grounding.py`)
     finds 3 real archetype shapes (PHONO-dominant, ORTHO-dominant
     high-density, mild/SEG-leaning) — evidence the simulator's assumed
     archetypes have real counterparts.
- **ASSISTments 2009-10 skill builder** (525k responses, 4,217 students;
  cite Feng, Heffernan & Koedinger 2009): real longitudinal per-student
  math data. `profiling/assistments_profiling.py` finds 2 stable behavioral
  clusters (coping: 72% accuracy vs struggling: 41% accuracy, 3.4x hint
  rate, faster/impulsive responses) with split-half stability ARI=0.587 on
  1,519 students — real-data evidence that difficulty profiles are stable
  per-student traits, the empirical precondition for H3.
- **`profiling/experiment_ari_sweep.py`** — systematic 2-factor experiment
  (archetype separation x observations-per-student), 20 cells x 3 seeds ->
  `data/profiling/ari_sweep_results.csv`, ready for a Ch.5 heatmap. Headline:
  at weak separation (d=2), even 600 obs/student only reaches ARI 0.68; at
  d>=8, 200 obs/student suffice for ARI>0.9.

## Pilot prototype (2026-07-05)

`app/app.py` — Flask web prototype wrapping the trained modules end-to-end:
student view (`/aluno/<nome>`: writing task in English + subtraction +
fractions per round) and teacher dashboard (`/professor`: per-student error
profiles, dominant pattern, cross-session persistence flag). SQLite storage,
local-only by design (GDPR posture for a school pilot). Run from `System/`:
`python app/app.py`. Detection uses `linguistic/final_model` with a
confidence threshold (`DETECT_CONF`, default 0.55) — raw argmax was too
trigger-happy for teacher-facing use; punctuation-aware tokenization fixed
systematic sentence-final SEG false positives. `demo.py` is the CLI
equivalent for quick testing.

`math/experiment_distinguishability.py` — diagnostic item design result
(now in Ch.5): 2-digit problems can NEVER distinguish all three subtraction
malrules (0%); 6-digit with forced zero-borrow distinguishes all three on
74.8% of items. Item difficulty is a prerequisite for diagnosis, not an
obstacle. Results: `data/math/distinguishability_results.csv`.

## Screening mode (2026-07-05, later)

`screening/` — active-screening layer motivated by the domain gap:
- `item_bank.py` — diagnostic items: ORTHO-trap / PHONO-transparent /
  SEG-compound word families + math items admitted only when all malrules
  are pairwise distinguishable (5-digit, forced zero-borrow).
- `scorer.py` — closed-world deterministic scorer (target known → taxonomy
  mapper, no neural model, no domain gap). Note the deliberate routing
  difference vs free text: in dictation, real-word outcomes ('blent' for
  'blend') ARE spelling errors. Self-test 8/8.
- `experiment_profile_fidelity.py` — **key result**: on Holbrook, open-world
  (neural) vs closed-world per-student profiles have mean cosine 0.882, BUT
  Spearman rank correlation is 0.898 for error DENSITY vs only 0.311/0.179/0.618
  for PHONO/ORTHO/SEG proportions. Domain shift preserves who-errs-a-lot,
  destroys who-errs-how → type-resolved profiling of the target population
  should come from screening. Results: `data/profiling/profile_fidelity_results.json`.
- App: `/rastreio/<nome>` route + teacher dashboard screening table with
  z-score flags vs class norm.

## Next steps

1. Run real linguistic training on the RTX 5070 (`RUN_ON_GPU.md`) — now with
   the 1M synthetic + 6.9k real combined corpus.
2. After training, evaluate `final_model` on the two held-out benchmarks:
   `fce_test_bio.json` (same-domain) and `holbrook_bio.json` (domain shift:
   struggling children) — the gap between them is a citable Ch.5 result.
3. Extend the math ML-vs-symbolic comparison to fractions and to real MaE examples.
4. Once real (non-smoke-test) numbers exist, revise `ch3/chapter3.tex` and
   `ch4/chapter4.tex` — implement first, write the definitive chapter text
   after (per agreed workflow).
