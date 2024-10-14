"""Calculates where are the predictions empty in a output file"""

import argparse
import numpy as np
from typing import List
from termcolor import colored


def find_empty_indexes(lines: List[str]) -> List[int]:
    """Finds the indexes of the empty predictions in the output file"""
    indexes = []
    for i, line in enumerate(lines):
        if line.strip().split("\t")[1] == line.strip().split("\t")[2]:
            indexes.append(i)
    print(len(indexes) / len(lines))
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
    args.add_argument("--k", type=int, default=0)
    args = args.parse_args()
    with open(args.outfile, "r") as f:
        lines = f.readlines()
    empty_indexes = find_empty_indexes([i.strip() for i in lines])
    print(empty_indexes)
    for idx in np.random.choice(empty_indexes, 5):
        print("\n".join(find_neighbours(idx, args.k, lines)))
        print("\n\n\n")
