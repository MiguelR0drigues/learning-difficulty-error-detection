"""Ch.5 experiment: how much data, and how distinct must profiles be, before
clustering reliably recovers archetypes?

Earlier ad-hoc finding (see thesis notes): weak separation gave ARI~0.53,
strong gave ~0.97. This script turns that into a systematic 2-factor sweep:

- dominance d: the dominant error type's weight vs 1.0 for all others
  (d=2 -> barely distinct profiles ... d=12 -> sharply distinct);
- observations per student: n_sessions x responses_per_session.

Each cell runs 3 seeds; we report mean/std of KMeans ARI. Output CSV is
ready for a heatmap plot in Ch.5. Interpretation for the thesis: this
quantifies the DEPLOYMENT REQUIREMENT of the profiling module -- how many
responses a real classroom rollout must collect per student (given an
assumed profile separability) before its clusters can be trusted.
"""
from __future__ import annotations

import csv
import random

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import simulate_and_cluster as sac


def make_archetypes(d: float) -> dict:
    L = sac.ALL_ERROR_LABELS
    def w(dom):
        return {k: (d if k in dom else 1.0) for k in L}
    return {
        "PHONO_DOMINANT": {"error_rate": 0.55, "weights": w(["PHONO"])},
        "SEG_DOMINANT": {"error_rate": 0.50, "weights": w(["SEG"])},
        "MATH_PROCEDURAL_DOMINANT": {"error_rate": 0.55, "weights": w(["SFL", "BORROW_ZERO", "BORROW_NO_DEC"])},
        "FRACTION_DOMINANT": {"error_rate": 0.50, "weights": w(["FRAC_ADD", "FRAC_COMPARE"])},
        "LOW_DIFFICULTY": {"error_rate": 0.12, "weights": {k: 1.0 for k in L}},
        "MIXED_HIGH_DIFFICULTY": {"error_rate": 0.65, "weights": {k: 1.0 for k in L}},
    }


def run_cell(d: float, n_sessions: int, rps: int, seed: int, n_students: int = 180) -> float:
    sac.ARCHETYPES = make_archetypes(d)
    rng = random.Random(seed)
    students = sac.simulate_students(n_students, n_sessions=n_sessions,
                                     responses_per_session=rps, rng=rng)
    X, _ = sac.featurize(students, responses_per_session=rps)
    ids = {n: i for i, n in enumerate(sac.ARCHETYPES)}
    y = np.array([ids[s.true_archetype] for s in students])
    Xs = StandardScaler().fit_transform(X)
    lab = KMeans(n_clusters=len(ids), n_init=10, random_state=seed).fit_predict(Xs)
    return adjusted_rand_score(y, lab)


if __name__ == "__main__":
    dominances = [2, 4, 8, 12]
    obs_grid = [(2, 5), (4, 10), (6, 10), (10, 20), (20, 30)]
    seeds = [0, 1, 2]

    rows = []
    print(f"{'d':>4} {'sessions':>8} {'resp/sess':>9} {'total_obs':>9} {'ARI_mean':>9} {'ARI_std':>8}")
    for d in dominances:
        for (ns, rps) in obs_grid:
            aris = [run_cell(d, ns, rps, s) for s in seeds]
            m, sd = float(np.mean(aris)), float(np.std(aris))
            rows.append({"dominance": d, "n_sessions": ns, "responses_per_session": rps,
                         "total_obs": ns * rps, "ari_mean": round(m, 4), "ari_std": round(sd, 4)})
            print(f"{d:>4} {ns:>8} {rps:>9} {ns*rps:>9} {m:>9.3f} {sd:>8.3f}")

    out = "../data/profiling/ari_sweep_results.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved {len(rows)} cells -> {out}")
