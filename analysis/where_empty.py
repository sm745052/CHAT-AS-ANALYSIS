"""Calculates where are the predictions empty in a output file"""

import argparse
import numpy as np
from typing import List
from termcolor import colored
import sys


import argparse
from typing import List, Tuple


def space_correction(x: str) -> str:
    x = x.replace(" ,", ",")
    x = x.replace(" .", ".")
    x = x.replace(" ?", "?")
    x = x.replace(" !", "!")
    x = x.replace(" ’", "’")
    return x


def midpoint_check(splitted_line: str) -> bool:
    if len(splitted_line.split("\t")) != 5:
        return False
    prefix, gt, _, _, _ = splitted_line.split("\t")
    utterance = space_correction(prefix + gt)
    utterance = utterance.strip("\n")
    if utterance[: len(utterance) // 2] == utterance[len(utterance) // 2 :]:
        return False
    return True


# FUNCTION_MAPPING_BY_MODEL = {
#     "t5": midpoint_check,
#     "gpt2": midpoint_check,
#     "phi.prompt.word10": midpoint_check,
#     "phi.finetune.word3": midpoint_check,
#     "gpt4": midpoint_check,
#     "mistral.prompt.word2": midpoint_check,
#     "renee": midpoint_check,
# }


def correct(splitted_lines: List[str]) -> Tuple[List[str], List[str]]:
    correct_lines = []
    wrong_lines = []
    for splitted_line in splitted_lines:
        if midpoint_check(splitted_line):
            correct_lines.append(splitted_line)
        else:
            wrong_lines.append(splitted_line)
    return correct_lines, wrong_lines


def find_empty_indexes(lines: List[str]) -> List[int]:
    """Finds the indexes of the empty predictions in the output file"""
    indexes = []
    for i, line in enumerate(lines):
        if(line.strip() == '####'):
            continue
        if len(line.strip().split("\t")[2]) == 0:
            indexes.append(i)
    return indexes


def find_neighbours(index: int, n: int, lines: List[str]) -> List[str]:
    """Finds the neighbours of the index with a window of n above and below"""
    above_neighbors = [
        f'{colored("{}:".format(i), "green")} {lines[i]}'
        for i in range(max(0, index - n), index)
    ]
    itself = f'{colored("{}:".format(index), "red")} {lines[index]}'
    below_neighbors = [
        f'{colored("{}:".format(i), "green")} {lines[i]}'
        for i in range(index + 1, min(len(lines), index + n + 1))
    ]
    return above_neighbors + [itself] + below_neighbors


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--outfile", type=str, default="outputs/out.seen.ddc.qb")
    args.add_argument("--k", type=int, default=2)
    args.add_argument('--context_correction', action='store_true')
    args = args.parse_args()
    with open(args.outfile, "r") as f:
        lines = f.readlines()
    if(args.context_correction):
        x = []
        for line in lines:
            if correct([line])[0]:
                x.append(line)
            else:
                x.append("####")
        lines = x
    empty_indexes = find_empty_indexes([i.strip() for i in lines])
    print(empty_indexes)
    for idx in np.random.choice(empty_indexes, 5):
        print("\n".join(find_neighbours(idx, args.k, lines)))
        print("\n\n\n")
