"""Parser for ERRANT M2 format (used by BEA-2019/CoNLL-14 gold annotations).

M2 format: each block starts with 'S <tokenized source sentence>' followed
by zero or more 'A <start> <end>|||<type>|||<replacement>|||...' edit lines.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class M2Edit:
    start: int
    end: int
    error_type: str
    replacement: str


@dataclass
class M2Sentence:
    tokens: list
    edits: list  # list[M2Edit], annotator 0 only (gold)


def parse_m2_file(path: str) -> list:
    sentences = []
    tokens = None
    edits = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if tokens is not None:
                    sentences.append(M2Sentence(tokens, edits))
                tokens = line[2:].split()
                edits = []
            elif line.startswith("A "):
                parts = line[2:].split("|||")
                start_end, error_type, replacement = parts[0], parts[1], parts[2]
                start, end = map(int, start_end.split())
                annotator_id = parts[-1]
                if error_type == "noop":
                    continue
                if annotator_id != "0":
                    continue  # keep only the first/gold annotator
                edits.append(M2Edit(start, end, error_type, replacement))
            elif line.strip() == "":
                pass
    if tokens is not None:
        sentences.append(M2Sentence(tokens, edits))
    return sentences


def edit_to_o_c_str(sent: M2Sentence, edit: M2Edit) -> tuple:
    o_str = " ".join(sent.tokens[edit.start:edit.end])
    c_str = edit.replacement
    return o_str, c_str
