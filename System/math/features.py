"""
Feature extraction for the "simple ML classifier" arm of the math error
detection module (Thesis Q3: "How effectively do symbolic methods and
simple ML classifiers identify characteristic arithmetic procedural bugs
... from typed numeric answers?"). This gives the classifier the same raw
information a symbolic method would use (the problem and the student's
answer), but as hand-built numeric features rather than a hard-coded
malrule simulation -- the point is to see whether a learned model matches,
beats, or falls short of the symbolic detector, and on what kind of cases.
"""
from __future__ import annotations

from fractions import Fraction


def subtraction_features(a: int, b: int, student_answer: int) -> dict:
    correct = a - b
    diff_from_correct = student_answer - correct
    a_digits = [int(c) for c in str(a)]
    b_digits = [int(c) for c in str(b).rjust(len(str(a)), "0")]
    ans_digits = [int(c) for c in str(student_answer).rjust(len(str(a)), "0")] if len(str(student_answer)) <= len(str(a)) else None

    borrow_needed = any(x < y for x, y in zip(a_digits[::-1], b_digits[::-1]))
    zero_in_minuend = "0" in str(a)[:-1]  # zero in a non-units position

    return {
        "num_digits": len(str(a)),
        "borrow_needed": int(borrow_needed),
        "zero_in_minuend": int(zero_in_minuend),
        "diff_from_correct": diff_from_correct,
        "abs_diff_from_correct": abs(diff_from_correct),
        "student_answer_digits": len(str(student_answer)),
        "student_gt_correct": int(student_answer > correct),
        "units_digit_a": a_digits[-1],
        "units_digit_b": b_digits[-1],
        "tens_digit_a": a_digits[-2] if len(a_digits) > 1 else 0,
        "tens_digit_b": b_digits[-2] if len(b_digits) > 1 else 0,
    }


def fraction_add_features(n1: int, d1: int, n2: int, d2: int, student_answer: Fraction) -> dict:
    correct = Fraction(n1, d1) + Fraction(n2, d2)
    mediant = Fraction(n1 + n2, d1 + d2)
    return {
        "n1": n1, "d1": d1, "n2": n2, "d2": d2,
        "same_denominator": int(d1 == d2),
        "student_num": student_answer.numerator,
        "student_den": student_answer.denominator,
        "matches_mediant": int(student_answer == mediant),
        "matches_correct": int(student_answer == correct),
        "student_minus_correct": float(student_answer - correct),
    }


def fraction_compare_features(n1: int, d1: int, n2: int, d2: int, student_answer: int) -> dict:
    correct = (Fraction(n1, d1) > Fraction(n2, d2)) - (Fraction(n1, d1) < Fraction(n2, d2))
    return {
        "n1": n1, "d1": d1, "n2": n2, "d2": d2,
        "same_numerator": int(n1 == n2),
        "same_denominator": int(d1 == d2),
        "student_answer": student_answer,
        "matches_correct": int(student_answer == correct),
        "num_diff": n1 - n2,
        "den_diff": d1 - d2,
    }
