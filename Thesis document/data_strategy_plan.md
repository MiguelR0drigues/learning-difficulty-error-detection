# Data Strategy Plan (Consolidated)

**Purpose**: single reference for where training/evaluation data comes from for each of the three thesis modules, before implementation starts. Supersedes the optimistic COPLE2/ASSISTments references currently in `ch3/chapter3.tex` (Section "Data Strategy"). Language constraint has been relaxed: **English is acceptable**, which unlocks the strongest public GEC/error-annotation resources.

---

## 1. Linguistic Error Detection Module

| Source | What it is | Access | License / restrictions | Role |
|---|---|---|---|---|
| **BEA-2019 (Write&Improve + LOCNESS)** | 3,600 learner essays (A1–C2) + 100 native essays, ~800k words, span-level error annotation (ERRANT-compatible) | Hugging Face `bea2019st/wi_locness` — direct download, no application | Non-commercial research use; must credit CECL (Université catholique de Louvain) for the LOCNESS portion; no third-party redistribution | **Primary real training/dev data** |
| **JFLEG** | 1,511 sentences, fluency-oriented gold corrections, multiple human references | GitHub `keisks/jfleg` and HF `jhu-clsp/jfleg` — fully open, no license friction | None beyond standard research use | Evaluation benchmark (small, use for held-out fluency checks, not training) |
| **CoNLL-2014 / NUCLE** | 57k training sentences, 1,312 test sentences, 28 error types, M² scorer | comp.nus.edu.sg — requires signing a license form + team registration (~2 day turnaround); also mirrored on HF `nusnlp/NUCLE` under the same terms | Free but requires paperwork (NUS standard academic license) | Optional larger training set if BEA-2019 alone proves insufficient |
| **FCE Corpus (Cambridge Learner Corpus)** | 2.9M words, ~10k essays, ~80 error types + demographics | Requires signed license agreement with University of Cambridge | Non-commercial research/education only | Stretch goal — skip unless the above prove insufficient, the licensing overhead isn't worth it early on |

**Known gap**: these are all L2 adult/teen English-learner corpora, annotated with general GEC error types (spelling, punctuation, word order, etc.), not the thesis's specific taxonomy (ERR-PHONO / ERR-ORTHO / ERR-SEG). Two consequences:
1. A mapping step is needed: mine the "spelling"-type edits from BEA-2019/NUCLE and reclassify them into phonological vs. orthographic using phonetic-similarity heuristics (e.g., soundex/metaphone distance between student token and gold token) — segmentation errors can be mined directly from insert/delete-whitespace edits.
2. Rule-based synthetic corruption (already planned in Cap.3 §Data Strategy) remains necessary to guarantee balanced coverage of each fine-grained category, especially since real corpora under-represent child-specific dysorthography patterns (they're adult L2 corpora, not K-12 native-speaker corpora). Real data anchors realism; synthetic data guarantees label balance.

## 2. Mathematical Error Detection Module

| Source | What it is | Access | License | Role |
|---|---|---|---|---|
| **MaE (math-misconceptions)** | 55 documented middle-school algebra misconceptions, 220 expert-validated diagnostic examples (correct + incorrect + explanation) | GitHub `nancyotero-projects/math-misconceptions`, mirrored on HF `nanote/algebra_misconceptions` | MIT — fully open | **Primary real seed/validation data** for misconception categories, esp. integer subtraction & fractions |
| **DeepMind mathematics_dataset** | Generator producing millions of *correct* school-level arithmetic/algebra Q&A pairs | `pip install mathematics_dataset` or GitHub `google-deepmind/mathematics_dataset` | Apache-2.0 — fully open | Base-problem generator: supplies the large volume of valid problems that the malrule injector (SFL, borrow-from-zero, borrow-no-decrement, frac-add, frac-compare) corrupts to produce labeled synthetic errors at scale |

**No gap here** — this combination (real validated misconceptions + open-source correct-problem generator + our own malrule scripts) is self-sufficient. No urgent need for ASSISTments/Eedi given the licensing/scale overhead they'd add; keep them as a "if we need more diversity later" option.

## 3. Student Difficulty Profiling Module

**No public dataset exists** for this — there is no open, longitudinal, per-student, error-labeled dataset (real classroom data of this kind is exactly what GDPR/ethics constraints block, as discussed earlier). This module's input isn't raw text anyway; it consumes the *outputs* of modules 1 and 2 (error-type counts per response, over time, per student).

**Strategy**: a simulation harness, not a dataset.
1. Define a small set of synthetic "profile archetypes" (e.g., predominantly-phonological, predominantly-segmentation, predominantly-fraction-misconception, mixed/low-difficulty).
2. For each synthetic student, sample multiple responses over simulated time steps from the error pools built in modules 1–2, biased toward that archetype's error distribution.
3. Run the clustering pipeline and check whether it recovers the known ground-truth archetypes (cluster recovery validation) — this mirrors the approach Gomes et al. (2020) used for math misconception clustering, and directly operationalizes H3.
4. Real validation (optional, later): a small ethics-approved pilot with a partner school/teacher (20–50 anonymized real responses across a few students) as an external check — not a training requirement, a bonus validation step if time and approval allow.

---

## Summary Table

| Module | Real data | Synthetic/generated data | Status |
|---|---|---|---|
| Linguistic | BEA-2019 (+ JFLEG for eval; NUCLE/FCE optional) | Rule-based corruption + LLM-based augmentation for taxonomy-specific categories | Ready to start |
| Mathematical | MaE dataset | DeepMind mathematics_dataset as base + malrule injector | Ready to start |
| Profiling | None available (ethics-gated) | Simulated archetypes from modules 1–2 outputs | Ready to start; real pilot is a stretch goal |

## Next Steps
1. Download BEA-2019 (HF) and MaE (HF) and inspect actual record format before writing any loader code.
2. Build the malrule injector on top of `mathematics_dataset`.
3. Build the phonological/orthographic corruption rules for the linguistic side.
4. Only then revise `ch3/chapter3.tex` §Data Strategy to match what was actually implemented.
