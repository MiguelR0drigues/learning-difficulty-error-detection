"""Ch.5 experiment: malrule distinguishability as a function of problem
structure (digit count and borrow pattern).

Motivation: during dataset generation we found SFL and BORROW_NO_DEC are
numerically IDENTICAL for 2-digit single-borrow subtractions -- an examiner
cannot tell those two misconceptions apart from the answer alone, no matter
how good the detector is. This experiment generalizes that observation:
for each problem structure, what fraction of problems lets us uniquely
identify each malrule from the student's answer? The answer is a statement
about DIAGNOSTIC ITEM DESIGN: which problems a teacher/system should pose
if the goal is to distinguish specific misconceptions.

Output: ../data/math/distinguishability_results.csv + printed table.
"""
from __future__ import annotations

import csv
import random
from itertools import combinations

from mal_rules import (all_subtraction_malrule_answers,
                        generate_subtraction_problem,
                        needs_borrow_through_zero)

N_PER_CELL = 5000
DIGITS = [2, 3, 4, 5, 6]
MALRULES = ["SFL", "BORROW_ZERO", "BORROW_NO_DEC"]


def run_cell(digits: int, force_zero_borrow: bool, rng: random.Random) -> dict:
    unique_counts = {m: 0 for m in MALRULES}
    present_counts = {m: 0 for m in MALRULES}
    pair_collisions = {f"{a}={b}": 0 for a, b in combinations(MALRULES, 2)}
    all_distinct = 0
    n = 0
    for _ in range(N_PER_CELL):
        try:
            a, b = generate_subtraction_problem(digits=digits, force_borrow=True,
                                                force_zero_borrow=force_zero_borrow, rng=rng)
        except RuntimeError:
            continue
        n += 1
        answers = all_subtraction_malrule_answers(a, b)
        mal_answers = {m: answers[m] for m in MALRULES if m in answers}
        for m, ans in mal_answers.items():
            present_counts[m] += 1
            clash = any(ans == other for lab, other in answers.items() if lab != m)
            if not clash:
                unique_counts[m] += 1
        for m1, m2 in combinations(MALRULES, 2):
            if m1 in mal_answers and m2 in mal_answers and mal_answers[m1] == mal_answers[m2]:
                pair_collisions[f"{m1}={m2}"] += 1
        if len(mal_answers) == len(MALRULES) and len(set(mal_answers.values())) == len(MALRULES) \
           and answers["CORRECT"] not in mal_answers.values():
            all_distinct += 1
    row = {"digits": digits, "zero_borrow_forced": force_zero_borrow, "n": n,
           "all_three_distinct_pct": round(100 * all_distinct / n, 1)}
    for m in MALRULES:
        pc = present_counts[m]
        row[f"unique_{m}_pct"] = round(100 * unique_counts[m] / pc, 1) if pc else None
        row[f"applicable_{m}_pct"] = round(100 * pc / n, 1)
    for k, v in pair_collisions.items():
        row[f"collide_{k}_pct"] = round(100 * v / n, 1)
    return row


if __name__ == "__main__":
    rng = random.Random(7)
    rows = []
    for zb in [False, True]:
        for d in DIGITS:
            rows.append(run_cell(d, zb, rng))

    hdr = ["digits", "zero_borrow_forced", "n", "all_three_distinct_pct",
           "unique_SFL_pct", "unique_BORROW_ZERO_pct", "unique_BORROW_NO_DEC_pct",
           "collide_SFL=BORROW_NO_DEC_pct"]
    print(f"{'dig':>3} {'0-borrow':>8} {'all3distinct%':>13} {'uniqSFL%':>8} "
          f"{'uniqBZ%':>7} {'uniqBND%':>8} {'SFL=BND%':>8}")
    for r in rows:
        print(f"{r['digits']:>3} {str(r['zero_borrow_forced']):>8} "
              f"{r['all_three_distinct_pct']:>13} {r['unique_SFL_pct']:>8} "
              f"{str(r['unique_BORROW_ZERO_pct']):>7} {str(r['unique_BORROW_NO_DEC_pct']):>8} "
              f"{r['collide_SFL=BORROW_NO_DEC_pct']:>8}")

    out = "../data/math/distinguishability_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {out}")
