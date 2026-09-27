"""
Trains and evaluates a "simple ML classifier" for math error detection,
and compares it against the symbolic malrule detector (mal_rules.py) --
directly answering Thesis Q3: "How effectively do symbolic methods AND
simple ML classifiers identify characteristic arithmetic procedural bugs
... from typed numeric answers?"

Two evaluation conditions:
  1. Clean: exact malrule outputs (what both methods were designed for).
  2. Noisy: the malrule answer perturbed by +/-1 on one digit, simulating
     a student who applies the buggy procedure AND makes an unrelated slip
     (a realistic scenario symbolic exact-matching cannot handle at all,
     since it requires an EXACT answer match by construction).
"""
from __future__ import annotations

import json
import random
from fractions import Fraction

import numpy as np
# Switched from GradientBoostingClassifier to HistGradientBoostingClassifier:
# with the dataset scaled up to 1M examples (from ~3k) per Miguel's request
# for a "considerable" dataset, plain GradientBoostingClassifier didn't even
# finish fitting 50k rows in 44s (its boosting loop is single-threaded and
# doesn't scale). HistGradientBoostingClassifier (sklearn's LightGBM-style
# histogram-binned implementation) fits 700k rows in ~1.5s -- same "simple
# ML classifier" story for Q3, just an implementation that can actually
# handle the new data volume.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from features import subtraction_features, fraction_add_features, fraction_compare_features
from mal_rules import detect_subtraction_error, detect_fraction_add_error, detect_fraction_compare_error


def load_subtraction_data():
    data = json.load(open("../data/math/subtraction_synthetic.json"))
    rows = []
    for row in data:
        a, b = row["operands"]
        student_answer = int(row["student_answer"])
        rows.append((a, b, student_answer, row["label"]))
    return rows


def load_fraction_data():
    data = json.load(open("../data/math/fraction_synthetic.json"))
    add_rows, compare_rows = [], []
    for row in data:
        n1, d1, n2, d2 = row["operands"]
        if row["problem_type"] == "fraction_add":
            student_answer = Fraction(row["student_answer"])
            add_rows.append((n1, d1, n2, d2, student_answer, row["label"]))
        else:
            student_answer = int(row["student_answer"])
            compare_rows.append((n1, d1, n2, d2, student_answer, row["label"]))
    return add_rows, compare_rows


def perturb_subtraction_answer(ans: int, rng: random.Random) -> int:
    """Simulate a student who applies a malrule AND makes an unrelated
    +/-1 slip on one digit -- a realistic noise model exact symbolic
    matching cannot handle by construction."""
    s = list(str(ans))
    pos = rng.randrange(len(s))
    delta = rng.choice([-1, 1])
    new_digit = (int(s[pos]) + delta) % 10
    s[pos] = str(new_digit)
    return int("".join(s))


def run_subtraction_experiment():
    print("\n" + "=" * 70)
    print("SUBTRACTION: symbolic vs ML classifier")
    print("=" * 70)
    rows = load_subtraction_data()
    rng = random.Random(0)

    X = [subtraction_features(a, b, ans) for a, b, ans, _ in rows]
    y = [label for *_, label in rows]
    feature_names = list(X[0].keys())
    X_arr = np.array([[x[k] for k in feature_names] for x in X])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test, rows_train, rows_test = train_test_split(
        X_arr, y_enc, rows, test_size=0.3, random_state=0, stratify=y_enc)

    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X_train, y_train)

    # Score everything on STRING labels (not the encoder's int space): the
    # symbolic method can emit "OTHER_ERROR", a class that never appears in
    # training data by construction (the generator only ever labels exact
    # malrule outputs), so it must be scoreable even though the ML
    # classifier's LabelEncoder never learned that class.
    all_possible_labels = list(le.classes_) + ["OTHER_ERROR"]

    def ml_predict_labels(X):
        return le.inverse_transform(clf.predict(X))

    # --- Clean evaluation ---
    y_test_labels = le.inverse_transform(y_test)
    y_pred_ml_labels = ml_predict_labels(X_test)
    y_pred_symbolic_labels = [detect_subtraction_error(a, b, ans) for a, b, ans, _ in rows_test]

    print("\n--- CLEAN test set (exact malrule answers) ---")
    print(f"ML classifier   accuracy={accuracy_score(y_test_labels, y_pred_ml_labels):.3f}  "
          f"macro-F1={f1_score(y_test_labels, y_pred_ml_labels, average='macro', labels=all_possible_labels, zero_division=0):.3f}")
    print(f"Symbolic method accuracy={accuracy_score(y_test_labels, y_pred_symbolic_labels):.3f}  "
          f"macro-F1={f1_score(y_test_labels, y_pred_symbolic_labels, average='macro', labels=all_possible_labels, zero_division=0):.3f}")

    # --- Noisy evaluation: perturb the test answers by +/-1 on a digit ---
    noisy_rows = [(a, b, perturb_subtraction_answer(ans, rng), label) for a, b, ans, label in rows_test]
    X_noisy = np.array([[subtraction_features(a, b, ans)[k] for k in feature_names]
                         for a, b, ans, _ in noisy_rows])
    y_noisy_true_labels = [label for *_, label in noisy_rows]
    y_pred_ml_noisy_labels = ml_predict_labels(X_noisy)
    y_pred_symbolic_noisy_labels = [detect_subtraction_error(a, b, ans) for a, b, ans, _ in noisy_rows]

    print("\n--- NOISY test set (malrule answer +/-1 digit slip) ---")
    print(f"ML classifier   accuracy={accuracy_score(y_noisy_true_labels, y_pred_ml_noisy_labels):.3f}  "
          f"macro-F1={f1_score(y_noisy_true_labels, y_pred_ml_noisy_labels, average='macro', labels=all_possible_labels, zero_division=0):.3f}")
    print(f"Symbolic method accuracy={accuracy_score(y_noisy_true_labels, y_pred_symbolic_noisy_labels):.3f}  "
          f"macro-F1={f1_score(y_noisy_true_labels, y_pred_symbolic_noisy_labels, average='macro', labels=all_possible_labels, zero_division=0):.3f}")
    print("(Symbolic method requires an EXACT answer match by construction, so it necessarily")
    print(" collapses toward OTHER_ERROR/misclassification under noise -- this is the core")
    print(" trade-off Q3 asks about: precision-on-exact-patterns vs. robustness-to-variation.)")

    print("\nFull classification report (ML classifier, clean test set):")
    print(classification_report(y_test_labels, y_pred_ml_labels, zero_division=0))

    return {
        "clean": {"ml_accuracy": float(accuracy_score(y_test_labels, y_pred_ml_labels)),
                   "ml_macro_f1": float(f1_score(y_test_labels, y_pred_ml_labels, average='macro', labels=all_possible_labels, zero_division=0)),
                   "symbolic_accuracy": float(accuracy_score(y_test_labels, y_pred_symbolic_labels)),
                   "symbolic_macro_f1": float(f1_score(y_test_labels, y_pred_symbolic_labels, average='macro', labels=all_possible_labels, zero_division=0))},
        "noisy": {"ml_accuracy": float(accuracy_score(y_noisy_true_labels, y_pred_ml_noisy_labels)),
                   "ml_macro_f1": float(f1_score(y_noisy_true_labels, y_pred_ml_noisy_labels, average='macro', labels=all_possible_labels, zero_division=0)),
                   "symbolic_accuracy": float(accuracy_score(y_noisy_true_labels, y_pred_symbolic_noisy_labels)),
                   "symbolic_macro_f1": float(f1_score(y_noisy_true_labels, y_pred_symbolic_noisy_labels, average='macro', labels=all_possible_labels, zero_division=0))},
    }


def run_fraction_experiments():
    print("\n" + "=" * 70)
    print("FRACTIONS: symbolic vs ML classifier")
    print("=" * 70)
    add_rows, compare_rows = load_fraction_data()
    results = {}

    for name, rows, feat_fn, detect_fn in [
        ("fraction_add", add_rows, fraction_add_features, detect_fraction_add_error),
        ("fraction_compare", compare_rows, fraction_compare_features, detect_fraction_compare_error),
    ]:
        X = [feat_fn(n1, d1, n2, d2, ans) for n1, d1, n2, d2, ans, _ in rows]
        y = [label for *_, label in rows]
        feature_names = list(X[0].keys())
        X_arr = np.array([[x[k] for k in feature_names] for x in X])
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test, rows_train, rows_test = train_test_split(
            X_arr, y_enc, rows, test_size=0.3, random_state=0, stratify=y_enc)
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(X_train, y_train)
        y_pred_ml = clf.predict(X_test)
        y_pred_symbolic_labels = [detect_fn(n1, d1, n2, d2, ans) for n1, d1, n2, d2, ans, _ in rows_test]
        y_pred_symbolic = le.transform(y_pred_symbolic_labels)

        acc_ml = accuracy_score(y_test, y_pred_ml)
        acc_sym = accuracy_score(y_test, y_pred_symbolic)
        print(f"\n--- {name} ---")
        print(f"ML classifier   accuracy={acc_ml:.3f}  macro-F1={f1_score(y_test, y_pred_ml, average='macro'):.3f}")
        print(f"Symbolic method accuracy={acc_sym:.3f}  macro-F1={f1_score(y_test, y_pred_symbolic, average='macro'):.3f}")
        results[name] = {"ml_accuracy": float(acc_ml), "symbolic_accuracy": float(acc_sym)}

    return results


if __name__ == "__main__":
    sub_results = run_subtraction_experiment()
    frac_results = run_fraction_experiments()

    with open("../data/math_ml_vs_symbolic_results.json", "w") as f:
        json.dump({"subtraction": sub_results, "fractions": frac_results}, f, indent=2)
    print("\nSaved comparison results to data/math_ml_vs_symbolic_results.json")
