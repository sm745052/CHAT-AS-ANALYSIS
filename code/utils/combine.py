# combine seen unseen data
import os
import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prune', type=str, default=None)
    args = args.parse_args()
    x = os.listdir('./outputs')
    seen = ['./outputs/'+i for i in x if i.startswith('out.seen.')]
    unseen = ['./outputs/'+i for i in x if i.startswith('out.unseen.')]
    if(args.prune):
        seen = [i for i in seen if args.prune in i]
        unseen = [i for i in unseen if args.prune in i]
    print(seen)
    print(unseen)
    for i in seen:
        if(i.replace("seen", "unseen") in unseen):
            with open(i, 'r') as f:
                lines = f.readlines()
            with open(i.replace("seen", "all"), 'w') as f:
                f.writelines(lines)
            with open(i.replace("seen", "unseen"), 'r') as f:
                lines = f.readlines()
            with open(i.replace("seen", "all"), 'a') as f:
                f.writelines(lines)
        else:
            print("skipping ", i)