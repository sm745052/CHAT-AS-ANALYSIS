import argparse
import os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, required=True)
    args = parser.parse_args()
    n=0
    time = 0
    with open(args.eval_file, 'r') as f:
        for query in f:
            try:
                n+=1
                query = query.strip()
                query = query.split('\t')
                if(query[-1]=='-'):
                    continue
                time += float(query[5])
            except Exception as e:
                print(e)
                print(query)
    # find average inference time
    avg_inf_time = time /n
    print('Average inference time: {:.6f}'.format(avg_inf_time))





if __name__ == '__main__':
    main()
