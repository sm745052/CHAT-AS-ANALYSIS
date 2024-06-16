import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--result_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Creating graphs for {} and {}".format(args.eval_file, args.result_file))


    #compute predicted length distribution and gt length distribution
    costs = []
    predlen_dist = {}
    gtlen_dist = {}
    c=0
    n=0
    with open(args.eval_file, 'r') as f:
        for query in f:
            try:
                n+=1
                query = query.strip()
                query = query.split('\t')
                if(query[-1]=='-'):
                    continue
                y = query[1]
                pred = query[2]
                predlen_dist[len(pred)] = predlen_dist.get(len(pred), 0) + 1
                gtlen_dist[len(y)] = gtlen_dist.get(len(y), 0) + 1
                if(float(query[4])<1e-6 or float(query[3]) > 1e9):
                    continue
                costs.append(float(query[3]) / float(query[4]))
            except Exception as e:
                print(e)
                print(query)
    # print(costs)
    # #plot cost distribution using plt
    plt.figure(figsize=(10, 5))
    plt.hist(costs, bins=200)
    plt.xlabel('cost')
    plt.ylabel('frequency')
    plt.title('cost distribution')
    plt.savefig(os.path.join(args.output_dir, 'cost_distribution.png'))
    plt.clf()
    # plt both distributions
    x = np.arange(0, 100)
    y = []
    for i in x:
        y.append(predlen_dist.get(i, 0))
    plt.bar(x, y, color='green')
    plt.title("{} predicted length distribution".format(args.eval_file))
    plt.xlabel("Predicted length")
    plt.ylabel("Count")
    plt.savefig(os.path.join(args.output_dir, 'predlen_distribution.png'))
    plt.clf()
    y = []
    for i in x:
        y.append(gtlen_dist.get(i, 0))
    plt.bar(x, y, color='green')
    plt.title("{} truth length distribution".format(args.eval_file))
    plt.xlabel("Ground truth length")
    plt.ylabel("Count")
    plt.savefig(os.path.join(args.output_dir, 'gtlen_distribution.png'))
    plt.clf()

    # plot thres vs triger rate and synctatic match using axis twin
    df = pd.read_csv(args.result_file, sep=';')
    df = df[df['trigger_rate']!=0]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df['thres'], df['trigger_rate'], 'g-')
    ax2.plot(df['thres'], df['synctatic_match'], 'b-')
    ax1.set_xlabel('thres')
    ax1.set_ylabel('trigger_rate', color='g')
    ax2.set_ylabel('synctatic_match', color='b')
    ax1.set_title('{} Thres vs Trigger Rate and Synctatic Match'.format(args.result_file))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'thres_trigger_synctatic.png'))
    plt.clf()

    #plot of TR vs SM

    fig, ax = plt.subplots()
    ax.plot(df['trigger_rate'], df['synctatic_match'])
    ax.set_xlabel('trigger_rate')
    ax.set_ylabel('synctatic_match')
    ax.set_title('{} Trigger Rate vs Synctatic Match'.format(args.result_file))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'trigger_synctatic.png'))
    plt.clf()

    thrs = df['thres']
    tr = df['trigger_rate']
    sm = df['synctatic_match']

    # find AUC of TR vs SM
    auc_score = auc(tr, sm)
    with open(os.path.join(args.output_dir, 'auc.txt'), 'w') as f:
        f.write("{}".format(auc_score))



if __name__ == '__main__':
    main()
