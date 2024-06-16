from string import punctuation
import sys
import argparse
import random
import io
import os
import tqdm
import thres_ckecks
"""
take queries (x), gt (y), predictions (pred), costs (cost) and subword_len(if not like QB, then put 1) from stdin
then filter output based on threshold

"""




mapping = {
    'QB': (thres_ckecks.thres_checkQB),
    'MPC': (thres_ckecks.thres_checkMPC),
    'GPT2': (thres_ckecks.thres_checkGPT2),
    'NGAME': (thres_ckecks.thres_checkNGAME),
    'GPT4': (thres_ckecks.thres_checkGPT4),
    # 0.05 intervel from 0 to 0.9, 0.0005 intervel from 0.9 to 1
    'rBERT': (thres_ckecks.thres_checkNGAME),
}



def range_eval(l, r, input_, model, threshold):
    missed = 0
    exact = 0
    prefix_exact = 0
    matched_length = 0
    prefix_matched_length = 0
    times = 0
    tot_lines = 0
    
    args_input = input_
    args_model = model
    thres_ckeck = mapping[args_model]
    with open(args_input, 'r') as f:
        for query in tqdm.tqdm(f):
            query = query.strip()
            if(query.split('\t')[2] =='-' or len(query.split('\t')[2])<l or len(query.split('\t')[2])>r):
                continue
            tot_lines += 1
            try:
                x, y, pred, cost, subword_len, time = query.split('\t')
                # remove EOU temporary fix for GPT2 sloss without context
                pred = pred.replace("<|EOU|>", "")
                # NLG tokenizer temporary fix
                if(model == "GPT2"):
                    #remove single space before punctuation
                    y = y.replace(" ,", ",")
                    y = y.replace(" .", ".")
                    y = y.replace(" ?", "?")
                    y = y.replace(" !", "!")
                
            except Exception as e:
                print(e)
                print(query)
            if(query.split('\t')[-1] =='-'):
                missed += 1
                continue
            times += float(time)
            if thres_ckeck(x, y, pred, cost, subword_len, threshold):
                # check without case TO_DO
                y = y.lower()
                pred = pred.lower()
                if pred == y:
                    exact += 1
                    matched_length += len(y)
                if pred==y or y.startswith(pred) and len(pred)>1:
                    prefix_exact += 1
                    prefix_matched_length += len(pred)
            else:
                missed += 1
    return_dict = {
        "missed": missed,
        "exact": exact,
        "prefix_exact": prefix_exact,
        "matched_length": matched_length,
        "prefix_matched_length": prefix_matched_length,
        "times": times,     # includes wrong predictions also
        "tot_lines": tot_lines
    }
    return return_dict

def main():
    ranges = [(1, 5), (6, 10), (11, 20), (21, 1e9)]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--thres', type=float, required = True)

    args = parser.parse_args()

    with open(args.output, 'w') as f:
        # columns are the ranges
        # rows are MR and inference time
        f.write('metric;'+';'.join(["{}".format(i) for i in ranges]) + '\n')
        rows = [["" for i in ranges] for i in range(2)]
        for i, (l, r) in enumerate(ranges):
            return_dict = range_eval(l, r, args.input, args.model, args.thres)
            missed = return_dict["missed"]
            exact = return_dict["exact"]
            prefix_exact = return_dict["prefix_exact"]
            matched_length = return_dict["matched_length"]
            prefix_matched_length = return_dict["prefix_matched_length"]
            times = return_dict["times"]
            tot_lines = return_dict["tot_lines"]
            if(tot_lines - missed==0):
                continue
            # match rate
            rows[0][i] = str(exact/(tot_lines - missed))
            # inference time
            rows[1][i] = str(times/(tot_lines))
        f.write("MR;" + ";".join(rows[0]) + '\n')
        f.write("time;" + ";".join(rows[1]) + '\n')
            

if __name__ == '__main__':
    main()