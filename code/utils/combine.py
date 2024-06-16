# combine seen unseen data
import os


if __name__ == '__main__':
    x = os.listdir('./outputs')
    seen = ['./outputs/'+i for i in x if i.startswith('out.seen.')]
    unseen = ['./outputs/'+i for i in x if i.startswith('out.unseen.')]
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