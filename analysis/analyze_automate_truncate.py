import argparse
import os
import pandas as pd
import fnmatch

def model_map(x):
    if(x=='mpc' or x=='mpc.suffix' or 'gpt4' in x):
        return "MPC"
    if(x=='qb'):
        return "QB"
    if('gpt2' in x):
        return "GPT2"
    if('t5' in x):
        return "T5"
    if('phi.prompt' in x):
        return "PHI_PROMPT"
    if("phi.finetune" in x):
        return "PHI_FINETUNE"
    if("mistral" in x):
        return "MISTRAL"
    if(x=="ngame"):
        return "NGAME"
    if(x=='renee'):
        return "RENEE"
    print("failed ! model was not found for: ", x)
    raise Exception("model not found")


no_graphs = ['mpc', 'mpc.suffix', 'gpt4', 'renee']

TRUNCATION = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def execute(ob):
    for truncate in TRUNCATION:
        csv_file = ob['outfile'].replace("out.", "result.")+f".truncate_{truncate}.csv"
        csv_file = csv_file.replace("outputs", "results")
        os.system("python code/eval/eval_n.py --input {} --output {} --model {} --truncate {} --only_max".format(ob['outfile'], csv_file, model_map(ob['model']), truncate))


if __name__ =='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--file', type=str, default='na')
    args.add_argument("--multi", action='store_true')
    args.add_argument("--cores", type=int, default=4)
    args.add_argument("--prune", type = str, default='na')
    args = args.parse_args()
    if(args.file == 'na'):
        # get all files startnig with out
        files = os.listdir('./outputs')
        out_files = ['outputs/' + f for f in files if f.startswith('out.') and not f.startswith('out.range.')]
        if(args.prune != "na"):
            pattern = args.prune
            out_files = [i for i in out_files if fnmatch.fnmatch(i, pattern)]
        ls = []
        for i in out_files:
            ls.append({
                "outfile": i,
                "data": i.split('.')[2],
                "type" : i.split('.')[1],
                "model" : '.'.join(i.split('.')[3:]),
            })
        print(ls)
    else:
        out_files = [args.file]
        ls = []
        for i in out_files:
            ls.append({
                "outfile": i,
                "data": i.split('.')[2],
                "type" : i.split('.')[1],
                "model" : '.'.join(i.split('.')[3:]),
            })
        print(ls)
    # ls = ls[:2]
    print("working on outfiles: ", out_files)
    if(args.multi):
        from multiprocessing import Pool
        p = Pool(args.cores)
        p.map(execute, ls)
    else:
        for i in ls:
            execute(i)