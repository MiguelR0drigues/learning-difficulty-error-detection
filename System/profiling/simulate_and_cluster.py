"""
Student difficulty profiling module (Thesis Ch.3/Ch.4, "Student Difficulty
Profiling: Clustering and Aggregation").

There is no public dataset for this (see data_strategy_plan.md): no open,
longitudinal, per-student, error-labeled corpus exists, for the same
GDPR/minors reasons real classroom data collection is gated in general.
This module's actual input is the OUTPUT of modules 1-2 (per-response
error-type labels), not raw text/numbers, so the right validation strategy
is a simulation harness: define known "profile archetypes", generate
multi-session synthetic students biased toward each archetype's error
distribution, then check whether unsupervised clustering recovers the known
ground truth. This directly operationalizes H3 and mirrors the validation
approach Gomes et al. (2020) used for math misconception clustering.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              adjusted_rand_score)
from sklearn.preprocessing import StandardScaler

# Error-type vocabulary this module aggregates over (union of modules 1-2's
# output labels; CORRECT is tracked too since accuracy rate is itself a
# useful profiling feature).
LINGUISTIC_LABELS = ["PHONO", "ORTHO", "SEG"]
MATH_LABELS = ["SFL", "BORROW_ZERO", "BORROW_NO_DEC", "FRAC_ADD", "FRAC_COMPARE"]
ALL_ERROR_LABELS = LINGUISTIC_LABELS + MATH_LABELS

# Archetypes: each maps error label -> relative weight (probability mass)
# of a response exhibiting that error, conditioned on the response NOT
# being correct. `error_rate` controls how often a response is wrong at
# all (this itself is a profiling-relevant feature: overall difficulty).
ARCHETYPES = {
    "PHONO_DOMINANT":     {"error_rate": 0.55, "weights": {"PHONO": 12, "ORTHO": 1, "SEG": 1,
                                                            "SFL": 1, "BORROW_ZERO": 1, "BORROW_NO_DEC": 1,
                                                            "FRAC_ADD": 1, "FRAC_COMPARE": 1}},
    "SEG_DOMINANT":       {"error_rate": 0.50, "weights": {"PHONO": 1, "ORTHO": 1, "SEG": 13,
                                                            "SFL": 1, "BORROW_ZERO": 1, "BORROW_NO_DEC": 1,
                                                            "FRAC_ADD": 1, "FRAC_COMPARE": 1}},
    "MATH_PROCEDURAL_DOMINANT": {"error_rate": 0.55, "weights": {"PHONO": 1, "ORTHO": 1, "SEG": 1,
                                                            "SFL": 10, "BORROW_ZERO": 8, "BORROW_NO_DEC": 8,
                                                            "FRAC_ADD": 1, "FRAC_COMPARE": 1}},
    "FRACTION_DOMINANT":  {"error_rate": 0.50, "weights": {"PHONO": 1, "ORTHO": 1, "SEG": 1,
                                                            "SFL": 1, "BORROW_ZERO": 1, "BORROW_NO_DEC": 1,
                                                            "FRAC_ADD": 12, "FRAC_COMPARE": 12}},
    "LOW_DIFFICULTY":     {"error_rate": 0.12, "weights": {k: 1 for k in ALL_ERROR_LABELS}},
    "MIXED_HIGH_DIFFICULTY": {"error_rate": 0.65, "weights": {k: 1 for k in ALL_ERROR_LABELS}},
}


@dataclass
class SyntheticStudent:
    student_id: str
    true_archetype: str
    session_error_rates: list = field(default_factory=list)  # per-session list of dict(label -> count)


def _sample_response_label(archetype: dict, rng: random.Random) -> str:
    if rng.random() > archetype["error_rate"]:
        return "CORRECT"
    labels = list(archetype["weights"].keys())
    weights = list(archetype["weights"].values())
    return rng.choices(labels, weights=weights, k=1)[0]


def simulate_students(n_students: int, n_sessions: int = 6, responses_per_session: int = 10,
                       rng: random.Random = None) -> list:
    rng = rng or random.Random(0)
    archetype_names = list(ARCHETYPES.keys())
    students = []
    for i in range(n_students):
        archetype_name = archetype_names[i % len(archetype_names)]
        archetype = ARCHETYPES[archetype_name]
        student = SyntheticStudent(f"S{i:04d}", archetype_name)
        # allow a mild temporal drift: error rate decreases slightly over
        # sessions (a crude "some learning happens" assumption), so the
        # profiling module also has a temporal-trend feature to compute.
        for s in range(n_sessions):
            drift = max(0.0, archetype["error_rate"] - 0.03 * s)
            session_archetype = {"error_rate": drift, "weights": archetype["weights"]}
            counts = {label: 0 for label in ALL_ERROR_LABELS}
            for _ in range(responses_per_session):
                label = _sample_response_label(session_archetype, rng)
                if label != "CORRECT":
                    counts[label] += 1
            student.session_error_rates.append(counts)
        students.append(student)
    return students


def featurize(students: list, responses_per_session: int = 10) -> tuple:
    """Per-student feature vector: PROPORTION of each error label among
    this student's errors, PLUS the overall error rate, plus the trend.

    Real-data finding from the first run of this module: LOW_DIFFICULTY and
    MIXED_HIGH_DIFFICULTY archetypes share the exact same relative error-type
    distribution (uniform across all 8 labels) by design -- they differ only
    in how OFTEN a response is wrong, not in WHICH errors dominate when it
    is. A features vector built purely from error-type proportions (summing
    to 1) is mathematically blind to that difference: two archetypes with
    the same shape but different overall difficulty are indistinguishable
    in proportion-space. This is exactly analogous to the SFL/BORROW_NO_DEC
    malrule collision found in the math module -- a real information-
    theoretic limit of the feature representation, not a code bug. Adding
    the overall error rate as an explicit feature (not just relative
    proportions) fixes it, and is fixed here rather than swept under the
    rug."""
    rows = []
    for student in students:
        n_sessions = len(student.session_error_rates)
        totals = {label: 0 for label in ALL_ERROR_LABELS}
        session_totals = []
        for counts in student.session_error_rates:
            session_total = sum(counts.values())
            session_totals.append(session_total)
            for label, c in counts.items():
                totals[label] += c
        grand_total = sum(totals.values()) or 1
        rates = [totals[label] / grand_total for label in ALL_ERROR_LABELS]
        overall_error_rate = grand_total / (n_sessions * responses_per_session)
        x = np.arange(n_sessions)
        slope = np.polyfit(x, session_totals, 1)[0] if n_sessions > 1 else 0.0
        rows.append(rates + [overall_error_rate, slope])
    feature_names = ALL_ERROR_LABELS + ["overall_error_rate", "error_trend_slope"]
    return np.array(rows), feature_names


def run_clustering(n_students: int = 180, rng_seed: int = 42, n_sessions: int = 10, responses_per_session: int = 20):
    rng = random.Random(rng_seed)
    students = simulate_students(n_students, n_sessions=n_sessions, responses_per_session=responses_per_session, rng=rng)
    X, feature_names = featurize(students, responses_per_session=responses_per_session)
    true_labels = [s.true_archetype for s in students]
    true_label_ids = {name: i for i, name in enumerate(ARCHETYPES.keys())}
    y_true = np.array([true_label_ids[t] for t in true_labels])

    X_scaled = StandardScaler().fit_transform(X)

    results = {}

    # KMeans with the "correct" k (we know it here because it's synthetic;
    # in a real deployment you'd sweep k using silhouette/elbow).
    k = len(ARCHETYPES)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=rng_seed)
    km_labels = kmeans.fit_predict(X_scaled)
    results["kmeans"] = {
        "silhouette": silhouette_score(X_scaled, km_labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, km_labels),
        "adjusted_rand_index": adjusted_rand_score(y_true, km_labels),
    }

    # HDBSCAN: no fixed k, handles noise/outlier students natively.
    hdb = HDBSCAN(min_cluster_size=max(5, n_students // (2 * k)))
    hdb_labels = hdb.fit_predict(X_scaled)
    n_noise = int(np.sum(hdb_labels == -1))
    non_noise_mask = hdb_labels != -1
    hdb_metrics = {"n_clusters_found": len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0),
                    "n_noise_points": n_noise,
                    "adjusted_rand_index": adjusted_rand_score(y_true, hdb_labels)}
    if non_noise_mask.sum() > k and len(set(hdb_labels[non_noise_mask])) > 1:
        hdb_metrics["silhouette_excl_noise"] = silhouette_score(X_scaled[non_noise_mask], hdb_labels[non_noise_mask])
    results["hdbscan"] = hdb_metrics

    return students, X, feature_names, true_labels, km_labels, hdb_labels, results


if __name__ == "__main__":
    students, X, feature_names, true_labels, km_labels, hdb_labels, results = run_clustering()
    print(f"Simulated {len(students)} students across {len(ARCHETYPES)} archetypes")
    print(f"Feature vector: {feature_names}")
    print("\n=== KMeans ===")
    for k, v in results["kmeans"].items():
        print(f"  {k}: {v:.3f}")
    print("\n=== HDBSCAN ===")
    for k, v in results["hdbscan"].items():
        print(f"  {k}: {v}")

    print("\nInterpretation: Adjusted Rand Index close to 1.0 means the clustering")
    print("essentially recovered the known archetypes from error-rate patterns alone")
    print("(no student ever sees an explicit archetype label) -- direct evidence for H3.")
