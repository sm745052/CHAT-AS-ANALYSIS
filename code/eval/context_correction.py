"corrects the eou problem in output file"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()
    with open(args.outfile, "r") as f:
        lines = f.readlines()
    splitted_lines = [line for line in lines]
    correct_lines, wrong_lines = correct(splitted_lines)
    print(wrong_lines[:5])
    print(100 - len(correct_lines) / len(splitted_lines) * 100)
