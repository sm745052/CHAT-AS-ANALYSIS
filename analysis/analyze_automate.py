import argparse
import os
import pandas as pd

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
    print("failed ! model was not found for: ", x)
    raise Exception("model not found")


no_graphs = ['mpc', 'mpc.suffix', 'gpt4']

def execute(ob):
    csv_file = ob['outfile'].replace("out.", "result.")+".csv"
    csv_file = csv_file.replace("outputs", "results")
    graph_folder = ob['outfile'].replace("out.", "").replace(".", "_").replace("outputs", "results")
    print("graph folder is ", graph_folder)
    if(not os.path.exists(graph_folder)):
        os.makedirs(graph_folder)
    os.system("python code/eval/eval_n.py --input {} --output {} --model {}".format(ob['outfile'], csv_file, model_map(ob['model'])))
    if(ob['model'] not in no_graphs):
        os.system("python analysis/graphs.py --eval_file {} --result_file {} --output_dir {}".format(ob['outfile'], csv_file, graph_folder))
    df = pd.read_csv(csv_file, sep=';')
    # get the row with max f1
    # max_f1_row = df[df['F1'] == df['F1'].max()]
    max_trigger_rate_row = df[df['trigger_rate'] == df['trigger_rate'].max()]
    thres = max_trigger_rate_row['thres'].values[0]
    with open(graph_folder+"/best_thres.txt", 'w') as f:
        f.write("{}".format(thres))
    print("best thres for ", ob['outfile'], " is ", thres)


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
            out_files = [f for f in out_files if args.prune in f]
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
    if(args.multi):
        from multiprocessing import Pool
        p = Pool(args.cores)
        p.map(execute, ls)
    else:
        for i in ls:
            execute(i)