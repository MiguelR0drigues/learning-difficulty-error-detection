"""Grounds the profiling module's linguistic archetypes in REAL data.

Input: data/profiling/holbrook_student_profiles.json -- per-student error
counts from the Holbrook corpus (19 real struggling secondary-school
children; see linguistic/holbrook_to_bio.py for provenance).

Previously the simulation archetypes (simulate_and_cluster.ARCHETYPES) were
assumed. Here we (1) cluster the real students in error-proportion space,
(2) report which archetype shapes actually occur, and (3) export data-derived
archetype weights for the simulator. n=19 is far too small to train anything,
but it is enough to answer "are the assumed archetype SHAPES real?" -- and
notably it shows real students do differ in dominant error type (e.g. one
child is strongly PHONO-dominant, another strongly ORTHO-dominant).
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

LABELS = ["ERR-PHONO", "ERR-ORTHO", "ERR-SEG"]


def load_profiles(path="../data/profiling/holbrook_student_profiles.json"):
    return json.load(open(path))


def featurize(profiles: dict):
    names, rows = [], []
    for student, p in sorted(profiles.items()):
        total = sum(p[l] for l in LABELS)
        if total < 5:
            continue  # too few in-scope errors to estimate a distribution
        props = [p[l] / total for l in LABELS]
        density = total / max(p["n_tokens"], 1)  # in-scope errors per token
        names.append(student)
        rows.append(props + [density])
    return names, np.array(rows), LABELS + ["error_density"]


if __name__ == "__main__":
    profiles = load_profiles()
    names, X, feats = featurize(profiles)
    print(f"{len(names)} real students with >=5 in-scope errors; features: {feats}")

    Xs = StandardScaler().fit_transform(X)
    best = None
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        lab = km.fit_predict(Xs)
        sil = silhouette_score(Xs, lab)
        print(f"k={k}: silhouette={sil:.3f}")
        if best is None or sil > best[1]:
            best = (k, sil, lab, km)
    k, sil, lab, km = best
    print(f"\nBest k={k} (silhouette={sil:.3f})")

    archetypes = {}
    for c in range(k):
        members = [names[i] for i in range(len(names)) if lab[i] == c]
        centroid = X[lab == c].mean(axis=0)
        desc = {f: round(float(v), 4) for f, v in zip(feats, centroid)}
        dominant = max(LABELS, key=lambda l: desc[l])
        arch_name = f"REAL_{dominant.replace('ERR-','')}_C{c}"
        # weights usable by simulate_and_cluster (linguistic part only):
        # scale proportions so the max is ~10 (same magnitude convention).
        w = centroid[:len(LABELS)]
        w = (w / w.max() * 10).round(1)
        archetypes[arch_name] = {
            "members": members,
            "centroid": desc,
            "sim_weights": {l.replace("ERR-", ""): float(x) for l, x in zip(LABELS, w)},
            "error_density": desc["error_density"],
        }
        print(f"\nCluster {c} ({arch_name}): {members}")
        print(f"  centroid: {desc}")

    out = {"n_students": len(names), "k": k, "silhouette": round(float(sil), 3),
           "archetypes": archetypes,
           "provenance": "Holbrook corpus via linguistic/holbrook_to_bio.py"}
    with open("../data/profiling/holbrook_derived_archetypes.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved -> ../data/profiling/holbrook_derived_archetypes.json")
