"This calculates the meterics as a distribution  wrt the prefix length of the query"
import sys
import argparse
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import context_correction
import pandas as pd

sys.path.append("code/eval")
from eval_n import (
    Bucket,
    preprocess,
    mapping,
    _trailing_punctuation_correction_models,
    _gt_empty_correction_models,
    has_context,
)

METRICS = [
    # "tr",
    "sm",
    "pr_maxtr",
    "rc_maxtr",
    "total_lines",
]

RENAME_METRICS = {
    "tr": "trigger_rate",
    "sm": "synctatic_match",
    "pr_maxtr": "precision",
    "rc_maxtr": "recall",
    "total_lines": "frequency",
}


def get_points_by_prefix_length(
    points: List[List[Any]],
) -> Dict[int, List[List[Any]]]:
    points_by_prefix_length = defaultdict(list)
    for point in points:
        prefix_length = len(point[0])
        points_by_prefix_length[prefix_length].append(point)
    return points_by_prefix_length


def calculate(args):
    with open(args.input, "r") as f:
        lines = f.readlines()
    lines = list(filter(lambda x: len(x.split("\t")) == 5, lines))
    if has_context(args.input):
        lines, _ = context_correction.correct(lines)
    if args.truncate:
        lines = [
            [i[0], i[1], truncate(i[2], args.truncate), i[3], i[4]]
            for i in lines
        ]
    thres_check, thresholds = mapping[args.model]
    processed_points = [preprocess(line, args) for line in lines]
    print(args.model)
    if (
        args.model in _trailing_punctuation_correction_models
        or args.model in _gt_empty_correction_models
    ):
        processed_points = list(filter(lambda x: not len(x[1]) == 0, processed_points))
    points_by_prefix_length = get_points_by_prefix_length(processed_points)
    buckets_by_prefix_length = {
        prefix_length: Bucket(
            [point for point in points if thres_check(*point, thresholds[-1])],
            total_lines=len(points),
            model=args.model,
        )
        for prefix_length, points in points_by_prefix_length.items()
        if prefix_length > 0
    }
    metrics_by_prefix_length = {}
    for prefix_length, bucket in buckets_by_prefix_length.items():
        metrics_by_prefix_length[prefix_length] = bucket.compute()
    df_cols = ["prefix_length"] + [
        RENAME_METRICS[metric] for metric in METRICS
    ]
    df_vals = [
        [prefix_length]
        + [
            metrics_by_prefix_length[prefix_length][metric]
            for metric in METRICS
        ]
        for prefix_length, bucket in buckets_by_prefix_length.items()
    ]
    df = pd.DataFrame(df_vals, columns=df_cols)
    df.sort_values(by=["prefix_length"], inplace=True)
    df.to_csv(args.output, index=False, sep=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    calculate(args)
