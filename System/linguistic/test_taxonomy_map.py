import errant
from taxonomy_map import extract_and_map

# 'their'->'there' is a homophone confusion, which the thesis (Ch.3)
# explicitly classifies under ERR-ORTHO (not ERR-PHONO), even though it
# sounds identical -- ERR-PHONO is for phoneme-grapheme conversion errors
# (mis-spellings), homophone swaps are a distinct orthographic sub-category.
TEST_SENTENCES = [
    ("The dog run to the pak everyday.", "The dog runs to the park every day.",
     {"pak": "ERR-ORTHO", "everyday.": "ERR-SEG"}),  # 'run'->'runs' is OTHER (verb tense)
    ("I wich I could go their.", "I wish I could go there.",
     {"wich": "ERR-PHONO", "their.": "ERR-ORTHO"}),
    # 'rite' is also a real dictionary word (a ceremony), so this is a
    # homophone confusion too (-> ERR-ORTHO), same reasoning as their/there.
    ("She is rite about the goverment plan.", "She is right about the government plan.",
     {"rite": "ERR-ORTHO", "goverment": "ERR-ORTHO"}),
    ("We recieved the seperate boxes.", "We received the separate boxes.",
     {"recieved": "ERR-PHONO", "seperate": "ERR-PHONO"}),
    ("He goed to school", "He went to school", {}),  # irregular verb -> OTHER, not our taxonomy
]


def main():
    annotator = errant.load("en")
    all_rows = []
    for orig, cor, expected in TEST_SENTENCES:
        mapped = extract_and_map(annotator, orig, cor)
        print(f"\nORIG: {orig}\nCOR:  {cor}")
        for m in mapped:
            print(f"  {m.o_str!r:15} -> {m.c_str!r:15} errant={m.errant_type:12} thesis={m.thesis_label:10} ({m.reason})")
            all_rows.append(m)
        for key, exp_label in expected.items():
            match = [m for m in mapped if m.o_str.strip() == key or m.o_str.strip().rstrip('.') == key.rstrip('.')]
            assert match, f"expected an edit for token {key!r} in sentence {orig!r}, found none"
            got = match[0].thesis_label
            status = "OK" if got == exp_label else "FAIL"
            print(f"  [{status}] check {key!r}: got={got} expected={exp_label}")
            assert got == exp_label, f"{key}: got {got}, expected {exp_label}"

    from collections import Counter
    print("\nOverall thesis-label distribution across all edits found:")
    print(dict(Counter(m.thesis_label for m in all_rows)))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
