'calculates the number of completed tasks by analyze_automate.py when ran with all files in the outputs directory'
import os
import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--p', action='store_true')
    args = args.parse_args()
    print([i for i in os.listdir('outputs') if 'result.'+'.'.join(i.split('.')[1:])+'.csv' not in os.listdir('results')])
    if args.p:
        print(f"completed tasks:{100 * len(list(filter(lambda x:x.endswith('.csv'), os.listdir('Presults')))) / len(os.listdir('Poutputs'))}%")
    else:
        print(f"completed tasks:{100 * len(list(filter(lambda x:x.endswith('.csv'), os.listdir('results')))) / len(os.listdir('outputs'))}%")
