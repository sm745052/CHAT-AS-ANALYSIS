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



def execute(ob):
    csv_file = ob['outfile'].replace("out.", "result.prefix.")+".csv"
    csv_file = csv_file.replace("outputs", "results")
    graph_folder = ob['outfile'].replace("out.", "").replace(".", "_").replace("outputs", "results")
    print("graph folder is ", graph_folder)
    if(not os.path.exists(graph_folder)):
        os.makedirs(graph_folder)
    os.system("python code/eval/eval_p.py --input {} --output {} --model {}".format(ob['outfile'], csv_file, model_map(ob['model'])))
    os.system("python analysis/prefix_graphs.py --input {} --output_dir {}".format(csv_file, graph_folder))


if __name__ =='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--file', type=str, default='na')
    args.add_argument("--multi", action='store_true')
    args.add_argument("--cores", type=int, default=4)
    args.add_argument("--prune", type = str, default='na')
    args.add_argument("--pb", action='store_true')
    args = args.parse_args()
    if(args.file == 'na'):
        # get all files startnig with out
        if(args.pb):
            files = os.listdir('./Poutputs')
            out_files = ['Poutputs/' + f for f in files if f.startswith('out.') and not f.startswith('out.range.')]
        else:
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