# System for Error Detection and Learning Difficulties

**Automatic Analysis of Student Responses in Primary and Secondary Education**

MSc dissertation, Master's in Artificial Intelligence Engineering (MEIA), ISEP — Polytechnic Institute of Porto.
Miguel Mendes Rodrigues · Porto, September 2026 · Supervisor: Doctor Luís Filipe de Oliveira Gomes

[Read the full thesis (PDF)](Thesis%20document/main.pdf)

## What this is

Teachers routinely triage error-ridden student text — short answers, essays, worksheets, numeric solutions — under real workload constraints, and recurring error patterns that signal a learning difficulty often go unnoticed until problems compound. This thesis builds and evaluates a text-only system that detects those patterns automatically and aggregates them into interpretable, non-clinical difficulty profiles for educators, under the practical constraint that annotated data for the target population (children with learning difficulties) is scarce.

The system has three independent modules sharing one design philosophy — detect specific, pedagogically meaningful error types rather than a generic "right/wrong" score:

- **Linguistic error detection** — a Transformer sequence-labeling model (DeBERTa-v3) flags phonological, orthographic, and segmentation errors, trained with a synthetic-to-real curriculum and an ERRANT-derived error taxonomy.
- **Math misconception detection** — an exact symbolic simulator of arithmetic procedural bugs (e.g. borrow-from-zero, fraction misconceptions) compared head-to-head with a feature-based ML classifier trained on the same generated data.
- **Difficulty profiling** — unsupervised clustering aggregates per-student error detections into interpretable behavioral profiles, validated against known ground truth in simulation and for stability on real student data.

A pilot-ready prototype ties the three together end to end, including a European Portuguese instantiation of a worksheet screening mode.

## Key findings

- On real learner text, the linguistic module reaches a strict span-level F1 of **0.642**.
- The synthetic-data hypothesis (H2) is **not supported**: a multi-seed paired comparison against real-only training from scratch shows synthetic pre-training at 10⁶ examples *lowers* real-data F1 by 2.5–7 points, and is at best neutral at 10⁴ examples. A scale ablation shows more synthetic volume helps in-domain performance while hurting real-data transfer — evidence points to a synthetic/authentic distribution mismatch, not data volume, as the limiting factor.
- On real writing by children with literacy difficulties, performance drops by roughly 20 F1 points, quantifying a domain gap that motivates a two-regime design (deterministic scoring for structured worksheet items, the neural model for free text).
- For arithmetic, the two math methods are complementary: on exact malrule answers both hit 100% accuracy, but on noisy input (a correct procedure applied with an unrelated slip) the symbolic method drops to 0% while the ML classifier retains **61.7%**.
- Difficulty profiles show strong recoverability with clear archetype separation (KMeans ARI ≈ 0.97) and split-half stability on 1,519 real students, though weak-signal conditions degrade recovery substantially (ARI ≈ 0.53–0.58).

The full quantitative results, methodology, and threats to validity are in Chapters 3–5 of the thesis; see the abstract below for the short version.

## Repository layout

```
Thesis document/   LaTeX source and compiled PDF (main.pdf / 1181734.pdf)
System/             The real implementation — linguistic, math, and profiling modules
Experimentation/     Early proof-of-concept (semantic-similarity heuristic), superseded by System/
DataDisclosure/      State-of-the-art survey and data-sourcing disclosure
```

Each module under `System/` has its own README with setup instructions, real results, and pointers to the training scripts (`System/README.md`, `System/RUN_ON_GPU.md`). Trained model checkpoints and the largest synthetic datasets are not included in this repository — they run into hundreds of megabytes each and are fully regenerable from the scripts and the (included) real/small synthetic data.

## Tech stack

Python · PyTorch & Hugging Face Transformers (DeBERTa-v3) · scikit-learn (Gradient Boosting) · ERRANT · spaCy · KMeans/HDBSCAN · LaTeX (biblatex/biber, ISEP MEIA thesis class)

## Abstract

This thesis addresses the automatic detection of error patterns linked to learning difficulties in typed student responses, under the practical constraint that annotated data for the target population is scarce. It proposes a modular, text-only architecture with three components: a Transformer-based sequence-labeling module for difficulty-linked linguistic errors (phonological, orthographic, segmentation), a hybrid symbolic and machine-learning module for arithmetic procedural bugs, and an unsupervised profiling module that aggregates detections into interpretable difficulty profiles.

Data scarcity is addressed with rule-based synthetic error generation staged in a synthetic-to-real training curriculum. Evaluation on real learner text yields a span-level F-score of 0.642 under a strict exact-span metric. The synthetic-data hypothesis is not supported: a multi-seed paired comparison against real-only training from scratch shows that synthetic pre-training with 10⁶ examples lowers the real-data F-score by 2.5–7 points and is at best neutral with 10⁴ examples, and a scale ablation shows that increasing synthetic volume improves in-domain performance while degrading real-data transfer.

On real writing by children with literacy difficulties, performance drops by roughly 20 F-score points, quantifying the domain gap that defines the main direction for future work. For mathematics, exact symbolic simulation and a feature-based classifier are complementary, trading precision on modeled bugs against robustness to input noise. Difficulty profiles are supported by three analyses: recoverability in simulation, archetype-like structure in the writing of 19 real struggling children, and split-half stability of behavioral profiles on 1,519 real students.

A pilot-ready prototype demonstrates the pipeline end to end. Throughout, the system is framed as providing non-clinical, formative information for educators, with claims sized to the evidence.

---

*This repository accompanies an academic dissertation submitted to ISEP (Instituto Superior de Engenharia do Porto). Shared as a portfolio piece; see [`Thesis document/`](Thesis%20document/) for the full text and [`System/`](System/) for the implementation.*
