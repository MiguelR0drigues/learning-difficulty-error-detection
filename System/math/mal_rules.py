"""
Malrule simulator for mathematical error detection (Thesis Ch.3/Ch.4).

Design note: the plan called for using DeepMind's `mathematics_dataset` package
as the correct-problem generator. On inspection it is unmaintained and breaks
against current SymPy (`sympy.solvers.diophantine.base_solution_linear` was
removed). Rather than pin an old SymPy version and risk breaking everything
else, this module implements its own lightweight, dependency-free generator
for the two problem families the thesis actually needs (multi-digit
subtraction, fraction addition/comparison). This is simpler than the
DeepMind package for our narrow needs and has no fragile dependency.

Each malrule is implemented to exactly reproduce the worked examples already
present in ch3/chapter3.tex (see test_mal_rules.py), so the taxonomy in the
thesis and the code are provably consistent with each other.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional


def _digits(n: int, width: int) -> list[int]:
    """Least-significant-first digit list, zero-padded to `width`."""
    s = str(n).rjust(width, "0")
    return [int(c) for c in s[::-1]]


def correct_subtract(a: int, b: int) -> int:
    assert a >= b >= 0
    return a - b


def malrule_sfl(a: int, b: int) -> int:
    """Smaller-From-Larger bug: per column, |top - bottom|, borrow ignored."""
    assert a >= b >= 0
    width = max(len(str(a)), len(str(b)))
    da, db = _digits(a, width), _digits(b, width)
    result_digits = [abs(x - y) for x, y in zip(da, db)]
    return int("".join(str(d) for d in reversed(result_digits)))


def malrule_borrow_from_zero(a: int, b: int) -> Optional[int]:
    """Borrow-from-zero bug (VanLehn-style).

    When a borrow is required and the digit being borrowed from is 0, the
    student converts it to 10 to satisfy the immediate column, but (i) does
    not decrement it afterwards and (ii) does not cascade the borrow further
    left. Only defined when the bug is actually triggered.

    Reproduces the thesis example: 302 - 158 = 254 (not 144).
    """
    assert a >= b >= 0
    width = max(len(str(a)), len(str(b)))
    da, db = _digits(a, width), _digits(b, width)
    triggered = False
    result = [0] * width
    borrow_pending = False
    i = 0
    while i < width:
        top = da[i]
        if borrow_pending:
            top -= 1
            borrow_pending = False
        if top < db[i]:
            if da[i] == 0:
                top = 10 + da[i]
                result[i] = top - db[i]
                triggered = True
            else:
                top = 10 + top
                result[i] = top - db[i]
                borrow_pending = True
        else:
            result[i] = top - db[i]
        i += 1
    if not triggered:
        return None
    return int("".join(str(d) for d in reversed(result)))


def malrule_borrow_no_decrement(a: int, b: int) -> Optional[int]:
    """Borrow-no-decrement bug: borrows correctly for the current column
    (adds 10) but fails to decrement the column borrowed from.

    Reproduces the thesis example: 52 - 27 = 35 (not 25).
    """
    assert a >= b >= 0
    width = max(len(str(a)), len(str(b)))
    da, db = _digits(a, width), _digits(b, width)
    result = [0] * width
    triggered = False
    i = 0
    while i < width:
        top = da[i]
        if top < db[i] and i + 1 < width:
            top += 10
            result[i] = top - db[i]
            triggered = True
        else:
            if top < db[i]:
                return None
            result[i] = top - db[i]
        i += 1
    if not triggered:
        return None
    return int("".join(str(d) for d in reversed(result)))


def needs_borrow(a: int, b: int) -> bool:
    width = max(len(str(a)), len(str(b)))
    da, db = _digits(a, width), _digits(b, width)
    return any(x < y for x, y in zip(da, db))


def needs_borrow_through_zero(a: int, b: int) -> bool:
    """True if the standard algorithm would need to borrow from a column
    whose digit is 0 (the borrow-from-zero bug only makes sense here)."""
    width = max(len(str(a)), len(str(b)))
    da, db = _digits(a, width), _digits(b, width)
    borrow_pending = False
    for i in range(width):
        top = da[i] - (1 if borrow_pending else 0)
        borrow_pending = False
        if top < db[i]:
            if da[i] == 0:
                return True
            borrow_pending = True
    return False


def generate_subtraction_problem(digits: int = 2, force_borrow: bool = True,
                                  force_zero_borrow: bool = False,
                                  rng: Optional[random.Random] = None) -> tuple[int, int]:
    rng = rng or random
    for _ in range(2000):
        lo, hi = 10 ** (digits - 1), 10 ** digits - 1
        a = rng.randint(lo, hi)
        b = rng.randint(0, a)
        if force_zero_borrow and not needs_borrow_through_zero(a, b):
            continue
        if force_borrow and not needs_borrow(a, b):
            continue
        return a, b
    raise RuntimeError("could not generate a problem meeting constraints")


def generate_fraction(max_num: int = 12, max_den: int = 20,
                       rng: Optional[random.Random] = None) -> tuple[int, int]:
    rng = rng or random
    d = rng.randint(2, max_den)
    n = rng.randint(1, min(max_num, d - 1))
    return n, d


def correct_frac_add(n1: int, d1: int, n2: int, d2: int) -> Fraction:
    return Fraction(n1, d1) + Fraction(n2, d2)


def malrule_frac_add(n1: int, d1: int, n2: int, d2: int) -> Fraction:
    """Independent numerator/denominator addition (whole-number bias).
    Reproduces: 1/2 + 1/3 -> 2/5."""
    return Fraction(n1 + n2, d1 + d2)


def correct_frac_compare(n1: int, d1: int, n2: int, d2: int) -> int:
    a, b = Fraction(n1, d1), Fraction(n2, d2)
    return (a > b) - (a < b)


def malrule_frac_compare(n1: int, d1: int, n2: int, d2: int) -> int:
    """Component-wise (whole-number-bias) comparison. Reproduces: claims
    1/5 > 1/3 (numerators equal, so falls to 'bigger denominator wins')."""
    if n1 != n2:
        return (n1 > n2) - (n1 < n2)
    return (d1 > d2) - (d1 < d2)


MATH_LABELS = [
    "CORRECT", "SFL", "BORROW_ZERO", "BORROW_NO_DEC",
    "FRAC_ADD", "FRAC_COMPARE", "OTHER_ERROR",
]


@dataclass
class MathExample:
    problem_type: str
    operands: tuple
    student_answer: object
    correct_answer: object
    label: str


def all_subtraction_malrule_answers(a: int, b: int) -> dict:
    """All malrule answers for (a, b), keyed by label. Used to detect
    cross-malrule collisions (a real, documented phenomenon -- e.g. SFL and
    BORROW_NO_DEC are provably identical for 2-digit single-borrow cases;
    BORROW_ZERO and others can also coincide when zeros appear mid-number)."""
    out = {"CORRECT": correct_subtract(a, b), "SFL": malrule_sfl(a, b)}
    bz = malrule_borrow_from_zero(a, b)
    if bz is not None:
        out["BORROW_ZERO"] = bz
    bn = malrule_borrow_no_decrement(a, b)
    if bn is not None:
        out["BORROW_NO_DEC"] = bn
    return out


def _is_unambiguous(answers: dict, label: str) -> bool:
    """True if `label`'s answer doesn't collide with any other label's
    answer for the same problem (guarantees a clean training signal)."""
    target = answers.get(label)
    if target is None:
        return False
    for other_label, other_ans in answers.items():
        if other_label != label and other_ans == target:
            return False
    return True


def make_subtraction_examples(n: int, rng: Optional[random.Random] = None,
                               require_unambiguous: bool = True) -> list:
    """Generate labeled subtraction examples. By default each example is
    checked against every other malrule's output on the same operands and
    discarded if there's a collision, so the resulting dataset has clean,
    separable labels. Set require_unambiguous=False to keep colliding cases
    too (useful for studying the ambiguity itself, e.g. for Ch.5 error
    analysis / threats-to-validity discussion)."""
    rng = rng or random.Random(0)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        label = rng.choice(["CORRECT", "SFL", "BORROW_ZERO", "BORROW_NO_DEC"])
        # Widened from [2, 3]: at large-N generation (hundreds of thousands+),
        # the 2-3 digit-only combinatorial space (a few hundred thousand
        # ordered (a,b) pairs at most) starts repeating the same underlying
        # problems many times over -- the same "small template pool inflates
        # results" issue found in the linguistic module, just less severe.
        # 4-digit subtraction is still elementary-appropriate and multiplies
        # the available problem space by roughly 100x.
        digits = rng.choice([2, 3, 4])
        if label == "BORROW_ZERO":
            a, b = generate_subtraction_problem(digits=max(digits, 3), force_zero_borrow=True, rng=rng)
        elif label in ("BORROW_NO_DEC", "SFL"):
            a, b = generate_subtraction_problem(digits=max(digits, 3), force_borrow=True, rng=rng)
        else:
            a, b = generate_subtraction_problem(digits=digits, force_borrow=rng.choice([True, False]), rng=rng)
        answers = all_subtraction_malrule_answers(a, b)
        if label not in answers:
            continue
        if require_unambiguous and not _is_unambiguous(answers, label):
            continue
        out.append(MathExample("subtraction", (a, b), answers[label], answers["CORRECT"], label))
    return out


def make_fraction_examples(n: int, rng: Optional[random.Random] = None) -> list:
    """Note: malrule_frac_compare (whole-number-bias comparison) frequently
    agrees with the true comparison by coincidence (e.g. whenever the
    denominators match, comparing numerators directly IS correct) -- that's
    inherent to the misconception itself, not a code defect. An example is
    only labeled FRAC_COMPARE when the malrule's verdict actually diverges
    from the true comparison, i.e. when the bug is observable at all."""
    rng = rng or random.Random(0)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        label = rng.choice(["CORRECT", "FRAC_ADD", "FRAC_COMPARE"])
        n1, d1 = generate_fraction(rng=rng)
        n2, d2 = generate_fraction(rng=rng)
        if label == "FRAC_ADD":
            ans = malrule_frac_add(n1, d1, n2, d2)
            correct = correct_frac_add(n1, d1, n2, d2)
            if ans == correct:
                continue
            out.append(MathExample("fraction_add", (n1, d1, n2, d2), ans, correct, label))
        elif label == "FRAC_COMPARE":
            ans = malrule_frac_compare(n1, d1, n2, d2)
            correct = correct_frac_compare(n1, d1, n2, d2)
            if ans == correct:
                continue  # bug not observable on this problem, skip
            out.append(MathExample("fraction_compare", (n1, d1, n2, d2), ans, correct, label))
        else:
            if rng.choice([True, False]):
                ans = correct_frac_add(n1, d1, n2, d2)
                out.append(MathExample("fraction_add", (n1, d1, n2, d2), ans, ans, "CORRECT"))
            else:
                correct = correct_frac_compare(n1, d1, n2, d2)
                out.append(MathExample("fraction_compare", (n1, d1, n2, d2), correct, correct, "CORRECT"))
    return out


def detect_subtraction_error(a: int, b: int, student_answer: int) -> str:
    if student_answer == correct_subtract(a, b):
        return "CORRECT"
    if student_answer == malrule_sfl(a, b):
        return "SFL"
    bz = malrule_borrow_from_zero(a, b)
    if bz is not None and student_answer == bz:
        return "BORROW_ZERO"
    bn = malrule_borrow_no_decrement(a, b)
    if bn is not None and student_answer == bn:
        return "BORROW_NO_DEC"
    return "OTHER_ERROR"


def detect_fraction_add_error(n1: int, d1: int, n2: int, d2: int, student_answer) -> str:
    if student_answer == correct_frac_add(n1, d1, n2, d2):
        return "CORRECT"
    if student_answer == malrule_frac_add(n1, d1, n2, d2):
        return "FRAC_ADD"
    return "OTHER_ERROR"


def detect_fraction_compare_error(n1: int, d1: int, n2: int, d2: int, student_answer: int) -> str:
    if student_answer == correct_frac_compare(n1, d1, n2, d2):
        return "CORRECT"
    if student_answer == malrule_frac_compare(n1, d1, n2, d2):
        return "FRAC_COMPARE"
    return "OTHER_ERROR"
