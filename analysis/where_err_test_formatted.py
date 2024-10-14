'finds the number of error lines in test formatted file. just returns the percent of lines without eou'
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    error = 0
    for line in lines:
        if '<eou>' not in line:
            # print(line)
            error += 1

    print(f'{error/total*100:.2f}')