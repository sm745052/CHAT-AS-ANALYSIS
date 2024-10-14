"""finds the number of lines in output files and prints them to the console"""

import argparse
import os
from typing import Dict, List


def count_lines(file):
    print(f"Counting lines in {file}")
    with open(file, "r") as f:
        return len(f.readlines())


def get_type(outfile):
    return outfile.split(".")[1]


def get_dataset(outfile):
    return outfile.split(".")[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--d", type=str, required=True)
    args = parser.parse_args()
    if args.file:
        print(f"Number of lines in {args.file}: {count_lines(args.file)}")
    else:
        print("Seeing all files in outputs directory")
        print("------------------------------------")
        files = list(filter(lambda x: args.d == get_dataset(x), os.listdir("outputs/")))
        lines_by_type: Dict[str, List[Dict[str, int]]] = {
            "seen": [],
            "unseen": [],
            "all": [],
        }
        for file in files:
            lines_by_type[get_type(file)].append(
                {"file": file, "lines": count_lines(f"outputs/{file}")}
            )
        for key in lines_by_type:
            print(f"Type: {key}")
            for file in lines_by_type[key]:
                print(f"{file['file']}: {file['lines']}")
            print("------------------------------------")
