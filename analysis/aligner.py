import os
import argparse
from termcolor import colored


def predict_raw_test_File(input_file):
    # print(input_file)
    input_file = input_file.lower()
    input_file = input_file.strip("\n")
    if "unseen" in input_file:
        if "cddc" in input_file:
            return "data/NcDDC/unseen/test_formatted.txt"
        if "cdstc7" in input_file:
            return "data/NcDSTC7/unseen/test_formatted.txt"
        if "ddc" in input_file:
            return "data/DDC/unseen/test_formatted.txt"
        if "dstc7" in input_file:
            return "data/DSTC7/unseen/test_formatted.txt"
        raise ValueError("Invalid input file")
    if "seen" in input_file:
        if "cddc" in input_file:
            return "data/NcDDC/seen/test_formatted.txt"
        if "cdstc7" in input_file:
            return "data/NcDSTC7/seen/test_formatted.txt"
        if "ddc" in input_file:
            return "data/DDC/seen/test_formatted.txt"
        if "dstc7" in input_file:
            return "data/DSTC7/seen/test_formatted.txt"
    if "all" in input_file:
        if "cddc" in input_file:
            return "data/NcDDC/all/test_formatted.txt"
        if "cdstc7" in input_file:
            return "data/NcDSTC7/all/test_formatted.txt"
        if "ddc" in input_file:
            return "data/DDC/all/test_formatted.txt"
        if "dstc7" in input_file:
            return "data/DSTC7/all/test_formatted.txt"
    raise ValueError("Invalid input file")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--outfile', type=str, required = True)
    args = args.parse_args()
    input_file = predict_raw_test_File(args.outfile)
    print(colored("test_formatted file: {}".format(input_file), "green"))
    with open(args.outfile, 'r') as f:
        predlines = f.readlines()
    with open(input_file, 'r') as f:
        testlines = f.readlines()
    if(abs(len(predlines) == len(testlines))<=1):
        exit(0)
    for i in range(min(len(predlines), len(testlines))):
        try:
            gt_test = testlines[i].strip("\n").split("\t")[1]
            pred_test = predlines[i].strip("\n").split("\t")[1]
        except Exception as e:
            print(colored("Error in line {}".format(i), "red"))
            print(colored("Pred: {}".format(predlines[i]), "red"))
            print(len(predlines[i].strip("\n").split("\t")))
            print(colored("GT: {}".format(testlines[i]), "green"))
            print(len(testlines[i].strip("\n").split("\t")))
            print(e)
            exit(0)
        if gt_test != pred_test:
            print(colored("Error in line {}".format(i), "red"))
            print(colored("Pred: {}".format(pred_test), "red"), colored("GT: {}".format(gt_test), "green"))
            exit(0)
