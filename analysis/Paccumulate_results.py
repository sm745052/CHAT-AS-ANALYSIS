import argparse
import pandas as pd



def get(args):
    if(args.null):
        df_tmp = pd.DataFrame({
            "trigger_rate": [""],
            "sm": [""],
            "re": [""],
            "pr": [""],
            "AUC": [""],
            "avg_pred_length": [""],
        })
        return df_tmp
    result_file = "Presults/result.{}.{}.{}.{}".format(args.type, args.data, args.model, "csv")
    graph_folder = "Presults/{}.{}.{}".format(args.type, args.data, args.model).replace(".", "_")
    print(result_file)
    df = pd.read_csv(result_file, sep=';')
    with open(graph_folder+"/best_thres.txt", 'r') as f:
        thres = f.read()
    try:
        with open(graph_folder+"/auc.txt", 'r') as f:
            auc = f.read()
    except:
        auc = "-"
    print("best thres for ", result_file, " is ", thres)
    df_row = df[df['thres'] == float(thres)]

    if(args.tr == 'max'):
        df_tmp = pd.DataFrame({
            "trigger_rate": [df_row['trigger_rate'].values[0]],
            "sm": [df_row['synctatic_match'].values[0]],
            "re": [df_row['partial_recall'].values[0]],
            "pr": [df_row['partial_precision'].values[0]],
            "AUC": [auc],
            "avg_pred_length": [df_row['avg_pred_length'].values[0]],
        })
    else:
        df_tmp = pd.DataFrame({
            "trigger_rate": [df_row['trigger_rate'].values[0]],
            "sm": [df_row['synctatic_match_100tr'].values[0]],
            "re": [df_row['partial_recall_100tr'].values[0]],
            "pr": [df_row['partial_precision_100tr'].values[0]],
            "AUC": [auc],
            "avg_pred_length": [df_row['avg_pred_length'].values[0]],
        })
    return df_tmp




def get_row(args):
    args.tr = 'max'
    args.type = "PB0"
    df1 = get(args)
    df1.columns = [i + " " + args.type + " " + args.tr for i in df1.columns]
    combined1 = df1
    args.type = "PB1"
    df2 = get(args)
    df2.columns = [i + " " + args.type + " " + args.tr for i in df2.columns]
    combined2 = df2
    args.type = "PB2"
    df3 = get(args)
    df3.columns = [i + " " + args.type + " " + args.tr for i in df3.columns]
    combined3 = df3
    combined = pd.concat([combined1, combined2, combined3], axis=1)
    return combined

if __name__ == '__main__':
    # format
    # unseen                                        seen                                            all
    # (sm, 100sm), (re, 100re), (pr, 100pr), (AUC)  (sm, 100sm), (re, 100re), (pr, 100pr), (AUC)    (sm, 100sm), (re, 100re), (pr, 100pr), (AUC)
    args = argparse.ArgumentParser()
    # args.add_argument('--data', type=str, default='ddc')
    # # args.add_argument('--type', type=str, default='seen')
    # args.add_argument('--model', type=str, default='qb')
    """
        QB
        MPC (char tries)
        MPC+Suffix
        T5 (12L+12L)
        GPT2 (12L)
        NGAME (K=300)
        NGAME (K=10)
        Mistral-7B (PromptEng)
        Phi-2 (2.7B) (PromptEng)
        Phi-2 (2.7B) (LoRA)
        GPT4
        QB
        MPC (char tries)
        MPC+Suffix
        T5 (12L+12L)
        GPT2 (12L)
        NGAME (TRIE)
        NGAME (K=10)
        Mistral-7B (PromptEng)
        Phi-2 (2.7B) (PromptEng)
        Phi-2 (LoRA)
        GPT4
    """
    args.add_argument('--d', type=str, default='ddc')
    args = args.parse_args()
    models = ['qb', 'mpc', 'mpc.suffix', 't5', 'gpt2.sloss', 'renee', 'mistral.prompt.word2', 'phi.prompt.word10', 'phi.finetune.word3', 'gpt4']
    data = [args.d, 'c' + args.d]
    todos = [(i, j) for i in data for j in models]
    df = -1
    args.null = False
    for i in todos:
        d = i[0]
        m = i[1]
        args.data = d
        args.model = m
        try:
            x = get_row(args)
        except Exception as e:
            print(e)
            args.null = True
            x = get_row(args)
            args.null = False
        if(type(df) == int):
            df = x
        else:
            df = pd.concat([df, x], axis=0)
        print(df)
    # set index
    index = []
    for i in todos:
        d = i[0]
        m = i[1]
        index.append(d + " " + m)
    df.index = index
    if(args.d == 'ddc'):
        df.to_csv("Pddc.csv", sep=';')
    else:
        df.to_csv("Pdstc7.csv", sep=';')