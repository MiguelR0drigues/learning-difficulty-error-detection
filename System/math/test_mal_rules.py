"""Validate the malrule simulator against the exact worked examples already
present in ch3/chapter3.tex, plus basic dataset-generation sanity checks."""
import random
from fractions import Fraction
from collections import Counter

from mal_rules import (
    correct_subtract, malrule_sfl, malrule_borrow_from_zero,
    malrule_borrow_no_decrement, correct_frac_add, malrule_frac_add,
    correct_frac_compare, malrule_frac_compare,
    make_subtraction_examples, make_fraction_examples,
    detect_subtraction_error, detect_fraction_add_error, detect_fraction_compare_error,
)


def check(name, got, expected):
    status = "OK" if got == expected else "FAIL"
    print(f"[{status}] {name}: got={got!r} expected={expected!r}")
    assert got == expected, f"{name} mismatch"


def main():
    print("=== Reproducing exact ch3/chapter3.tex worked examples ===")
    check("SFL 43-17", malrule_sfl(43, 17), 34)
    check("correct 43-17", correct_subtract(43, 17), 26)
    check("BORROW_ZERO 302-158", malrule_borrow_from_zero(302, 158), 254)
    check("correct 302-158", correct_subtract(302, 158), 144)
    check("BORROW_NO_DEC 52-27", malrule_borrow_no_decrement(52, 27), 35)
    check("correct 52-27", correct_subtract(52, 27), 25)
    check("FRAC_ADD 1/2+1/3", malrule_frac_add(1, 2, 1, 3), Fraction(2, 5))
    check("correct 1/2+1/3", correct_frac_add(1, 2, 1, 3), Fraction(5, 6))
    check("FRAC_COMPARE 1/5 vs 1/3", malrule_frac_compare(1, 5, 1, 3), 1)
    check("correct 1/5 vs 1/3", correct_frac_compare(1, 5, 1, 3), -1)

    print("\n=== Detection (reverse simulation) sanity checks ===")
    check("detect SFL", detect_subtraction_error(43, 17, 34), "SFL")
    check("detect BORROW_ZERO", detect_subtraction_error(302, 158, 254), "BORROW_ZERO")
    check("detect BORROW_NO_DEC (3-digit, unambiguous)", detect_subtraction_error(111, 2, 119), "BORROW_NO_DEC")
    check("detect CORRECT", detect_subtraction_error(52, 27, 25), "CORRECT")
    check("detect OTHER_ERROR", detect_subtraction_error(52, 27, 999), "OTHER_ERROR")
    check("detect FRAC_ADD", detect_fraction_add_error(1, 2, 1, 3, Fraction(2, 5)), "FRAC_ADD")
    check("detect FRAC_COMPARE", detect_fraction_compare_error(1, 5, 1, 3, 1), "FRAC_COMPARE")

    print("\n=== Known finding: SFL / BORROW_NO_DEC ambiguity for 2-digit numbers ===")
    assert malrule_sfl(52, 27) == malrule_borrow_no_decrement(52, 27) == 35
    print("Confirmed: malrule_sfl(52,27) == malrule_borrow_no_decrement(52,27) == 35")
    print("=> dataset generator now forces 3+ digits for BORROW_NO_DEC to stay separable from SFL.")

    print("\n=== Dataset generation sanity checks (n=500 each) ===")
    rng = random.Random(42)
    sub_examples = make_subtraction_examples(500, rng=rng)
    frac_examples = make_fraction_examples(500, rng=rng)
    print(f"subtraction examples generated: {len(sub_examples)}")
    print(f"fraction examples generated: {len(frac_examples)}")
    print("subtraction label distribution:", dict(Counter(e.label for e in sub_examples)))
    print("fraction label distribution:", dict(Counter(e.label for e in frac_examples)))

    mismatches = 0
    for e in sub_examples:
        a, b = e.operands
        pred = detect_subtraction_error(a, b, e.student_answer)
        if pred != e.label:
            mismatches += 1
            print("  MISMATCH:", e, "-> predicted", pred)
    print(f"subtraction round-trip mismatches: {mismatches}/{len(sub_examples)}")
    assert mismatches == 0

    frac_mismatches = 0
    for e in frac_examples:
        n1, d1, n2, d2 = e.operands
        if e.problem_type == "fraction_add":
            pred = detect_fraction_add_error(n1, d1, n2, d2, e.student_answer)
        else:
            pred = detect_fraction_compare_error(n1, d1, n2, d2, e.student_answer)
        if pred != e.label:
            frac_mismatches += 1
    print(f"fraction round-trip mismatches: {frac_mismatches}/{len(frac_examples)}")
    assert frac_mismatches == 0

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
