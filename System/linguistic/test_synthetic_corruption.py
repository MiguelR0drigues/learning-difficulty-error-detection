import random
from collections import Counter
from synthetic_corruption import generate_corpus, LABELS


def validate_bio(tags: list) -> bool:
    """A tag sequence is valid BIO if every I-X is immediately preceded by
    B-X or I-X (same X)."""
    prev = "O"
    for t in tags:
        if t.startswith("I-"):
            label = t[2:]
            if not (prev == f"B-{label}" or prev == f"I-{label}"):
                return False
        prev = t
    return True


def main():
    rng = random.Random(7)
    corpus = generate_corpus(300, rng=rng)
    print(f"generated {len(corpus)} examples")
    print("label distribution:", dict(Counter(r.label for r in corpus)))

    bad = 0
    for r in corpus:
        assert len(r.tokens) == len(r.bio_tags), f"length mismatch: {r}"
        if not validate_bio(r.bio_tags):
            bad += 1
            print("INVALID BIO:", r)
    print(f"invalid BIO sequences: {bad}/{len(corpus)}")
    assert bad == 0

    # every example must actually differ from its original sentence
    identical = sum(1 for r in corpus if r.original == r.corrupted)
    print(f"examples identical to original (should be 0): {identical}")
    assert identical == 0

    print("\n=== Sample outputs (5 per label) ===")
    for label in LABELS:
        print(f"\n--- {label} ---")
        examples = [r for r in corpus if r.label == label][:5]
        for r in examples:
            print(f"  orig: {r.original}")
            print(f"  corr: {r.corrupted}")
            print(f"  tags: {list(zip(r.tokens, r.bio_tags))}")
            print(f"  note: {r.note}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
