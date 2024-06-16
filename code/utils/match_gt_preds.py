import json
import os
import argparse
from tqdm import tqdm

def create_pred_map(args):
    """
    args.out_file: scrape text file extracted from the json scrapes 
    format -> prefix\tUNK(gt)\tprediction\tcost\tsub_word_count\n
    
    output: a map of the format {prefix: prediction}
    """
    out_file = args.inp_file
    pred_map = {}
    cnt = 0
    with open(out_file, "r") as f:
        for line in tqdm(f, desc="Creating Prediction Map"):
            cnt+=1 
            content = line.split("\t")
            # print(content)
            if(len(content)<5):
                print(cnt, content)
            prefix = content[0]
            prediction = content[2]
            cost = content[3]
            num_tok = content[4].strip('\n')
            
            # if prefix not in pred_map.values():
            pred_map[prefix] = [prediction, cost, num_tok]
                
    return pred_map

def get_gt(args):
    """
    input: test_formatted.txt (contains prefix\tgt)
    aim: need to iterate over the prefixs and if they match with a prefix in pred_map then return it as the output
    output: out.<type>.<dataset>.<model> file
    """
    pred_map = create_pred_map(args)
    # print("PRED MAP:", pred_map)
    # print(pred_map.keys())
    tst_file = args.tst_file
    out_dict = []
    cnt_present = 0
    cnt_total = 0
    with open(tst_file, "r") as f:
        for line in tqdm(f, desc="Processing Test Data"):
            cnt_total+=1
                
            prefix, gt = line.split("\t")
            # print(prefix)
            gt = gt.strip("\n")
            
            if prefix in pred_map:
                cnt_present+=1
                item = {
                    "prefix": prefix,
                    "gt": gt,
                    "prediction": pred_map[prefix][0],
                    "cost": pred_map[prefix][1],
                    "num_tok": pred_map[prefix][2]
                }
                out_dict.append(item)
    print("Percentage of prefix present in the scrapes: ", 100*(cnt_present/cnt_total))    
    return out_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_file', type=str, required=True, help='output_text file scraped from the output json')
    parser.add_argument('--tst_file', type=str, required=True, help='test_formatted.txt file of the dataset and split')
    parser.add_argument('--dataset', type=str, required=True, help='dataset used')
    parser.add_argument('--type', type=str, required=True, help='seen/unseen')
    parser.add_argument('--model', type=str, required=True, help='[gpt4, phi.finetune, phi.prompting, mistral]')
    args = parser.parse_args()
    
    out_dict = get_gt(args)
    
    print("Writing into outfile...") 
    
    with open(f"out.{args.type}.{args.dataset.lower()}.{args.model}", "w") as f:
        for data in out_dict:
            f.write(f'{data["prefix"]}\t{data["gt"]}\t{data["prediction"]}\t{data["cost"]}\t{data["num_tok"]}\n')
        
    with open(f"out.range.{args.type}.{args.dataset.lower()}.{args.model}", "w") as f:
        for data in out_dict:
            f.write(f'{data["prefix"]}\t{data["gt"]}\t{data["prediction"]}\t{data["cost"]}\t{data["num_tok"]}\t1\n')
   
    print("Writing complete...")
    
            
    