"""Profiling-module validation on REAL longitudinal data: ASSISTments
2009-10 skill-builder (525k responses, 4.2k students; CC-style public
release, cite Feng, Heffernan & Koedinger 2009).

The simulation harness (simulate_and_cluster.py) shows clustering recovers
KNOWN archetypes; the open question it can't answer is whether real
per-student behavioral profiles are STABLE enough to cluster at all. This
script answers that with a split-half stability test:

1. Per-student behavioral features (students with >= MIN_RESPONSES):
   accuracy, mean attempts, hint rate, median first-response time, and
   fraction of "mastery-fail" (>=3 attempts) responses.
2. Cluster all students (k swept by silhouette).
3. STABILITY: recompute the same features from each student's odd-indexed
   responses and even-indexed responses separately (two independent
   "half-histories"), cluster each half with the same k, and measure ARI
   between the two half-assignments. High ARI = a student's profile is a
   stable trait, not session noise -- the empirical precondition for
   difficulty profiling to be meaningful on real data.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

CSV_PATH = "../data/profiling_real/skill_builder_data.csv"
MIN_RESPONSES = 50


def load_responses(path=CSV_PATH):
    per_student = defaultdict(list)
    with open(path, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            try:
                per_student[row["user_id"]].append((
                    int(row["order_id"]),
                    int(row["correct"]),
                    int(row["attempt_count"]),
                    int(row["hint_count"]),
                    float(row["ms_first_response"]) if row["ms_first_response"] else np.nan,
                ))
            except (ValueError, KeyError):
                continue
    return {u: sorted(rs) for u, rs in per_student.items() if len(rs) >= MIN_RESPONSES}


def features(responses):
    a = np.array(responses, dtype=float)
    correct, attempts, hints, ms = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    ms = ms[~np.isnan(ms)]
    ms = ms[(ms > 0) & (ms < 300_000)]  # drop idle/timeout artifacts
    return [
        correct.mean(),
        attempts.mean(),
        (hints > 0).mean(),
        np.median(ms) / 1000 if len(ms) else 30.0,
        (attempts >= 3).mean(),
    ]


FEATURE_NAMES = ["accuracy", "mean_attempts", "hint_rate",
                 "median_first_response_s", "mastery_fail_rate"]


if __name__ == "__main__":
    students = load_responses()
    ids = sorted(students)
    print(f"{len(ids)} students with >={MIN_RESPONSES} responses")

    X = np.array([features(students[u]) for u in ids])
    Xs = StandardScaler().fit_transform(X)

    best = None
    for k in range(2, 7):
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
        sil = silhouette_score(Xs, lab)
        print(f"k={k}: silhouette={sil:.3f}")
        if best is None or sil > best[1]:
            best = (k, sil, lab)
    k, sil, full_labels = best
    print(f"Best k={k} (silhouette={sil:.3f})")

    print("\nCluster centroids (raw feature space):")
    for c in range(k):
        cent = X[full_labels == c].mean(axis=0)
        n = int((full_labels == c).sum())
        print(f"  C{c} (n={n}): " + ", ".join(f"{f}={v:.3f}" for f, v in zip(FEATURE_NAMES, cent)))

    # ---- split-half stability ----
    Xo = np.array([features(students[u][0::2]) for u in ids])
    Xe = np.array([features(students[u][1::2]) for u in ids])
    scaler = StandardScaler().fit(np.vstack([Xo, Xe]))
    lo = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(scaler.transform(Xo))
    le = KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(scaler.transform(Xe))
    ari = adjusted_rand_score(lo, le)
    print(f"\nSplit-half stability ARI (odd vs even half-histories, k={k}): {ari:.3f}")
    print("(high = profile is a stable per-student trait; low = session noise)")

    out = {"n_students": len(ids), "min_responses": MIN_RESPONSES,
           "feature_names": FEATURE_NAMES, "best_k": k,
           "silhouette": round(float(sil), 3),
           "split_half_ari": round(float(ari), 3),
           "cluster_sizes": {int(c): int((full_labels == c).sum()) for c in range(k)},
           "cluster_centroids_raw": {int(c): dict(zip(FEATURE_NAMES, X[full_labels == c].mean(axis=0).round(4).tolist())) for c in range(k)}}
    with open("../data/profiling/assistments_validation.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Saved -> ../data/profiling/assistments_validation.json")
