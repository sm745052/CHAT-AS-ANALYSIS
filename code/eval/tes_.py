import pandas as pd
import json

def space_correction(x):
    x = x.replace(" ,", ",")
    x = x.replace(" .", ".")
    x = x.replace(" ?", "?")
    x = x.replace(" !", "!")
    x = x.replace(" ’", "’")
    return x

def has_context(x):
    x = x.lower()
    if("cddc" in x or "cdstc7" in x):
        return True
    return False

def preprocess(i, model):
    i = i.lower()
    i = i.strip("\n")
    x, y = i.split('\t')
    x = x.split("<|eou|>")[-1]
    x = x.split("<eou>")[-1]
    x = x + "x"
    y = "x" + y
    y.replace("<|eou|>", "")
    y.replace("<eou>", "")
    x = x.strip()
    y = y.strip()
    x = x[:-1]
    y = y[1:]
    # NLG tokenizer temporary fix
    if(model == "GPT2" or "rBERT" in model):
        #remove single space before punctuation
        y = space_correction(y)
        x = space_correction(x)
    # print([x[-10:], y[:10]])
    # print([x, y])
    return [x, y]


def tes_compute(df_reduced, raw_tst_file, model):
    # raw test file contains the entire utterance (with context in case of context data)
    # df reduced contains only those prefixes for which we have predictions (based on threshold)
    # output: get saved_ks for each utterance and total TES metric value
    # print(raw_tst_file)
    saved_ks = []
    # map of prefix to predicted suffix
    map = df_reduced.set_index('prefix')['prediction'].to_dict()

    # print(map)
    unique_utterances = set() # set of unique utterances
    
    
    with open (raw_tst_file, "r") as f:
        for line in f:
            prefix, suffix = preprocess(line, model)
            unique_utterances.add(prefix + suffix) # add the new prefix and suffix to the set
    

    tess = []
    # print(unique_utterances)
    for utt in unique_utterances:
        c = 1
        s = utt[0]
        while(s!=utt):
            pred = map.get(s, "###############################")
            if(utt[len(s):].startswith(pred) and pred!=""):
                s += pred
            else:
                print(pred, "######", utt[len(s):])
                c += 1
                s += utt[len(s)]
        tess.append(1 - c/len(utt))
    tes_metric = sum(tess)/len(tess)
    print("TES Metric: ", tes_metric)

    return tes_metric

class Block:
    utterance = None
    ls = None

    def __init__(self, utterance, ls):
        self.utterance = utterance
        # print(utterance)
        self.ls = ls
        # print(ls)

def make_blocks(ls):
    blocks = []
    block = []
    utterance = ""
    for i in ls:
        present_utterance = i[0] + i[1]
        present_utterance = space_correction(present_utterance)
        if(present_utterance != utterance):
            if(len(block) > 0):
                blocks.append(Block(utterance, block))
            utterance = present_utterance
            block = []
        else:
            block.append(i)
    return blocks

def eval_block(block:Block, thres):
    # print(block.utterance)
    # print('\n'.join('####'.join(i) for i in block.ls))
    df = pd.DataFrame(block.ls, columns=["prefix", "gt", "prediction", "cost", "subword_len"])
    df['prefix'] = df['prefix'].apply(space_correction)
    df['gt'] = df['gt'].apply(space_correction)
    df['prediction'] = df['prediction'].apply(space_correction)
    dfr = df # TODO
    c = 1
    utt = space_correction(block.utterance)
    # print(utt)
    # print(len(utt))
    s = utt[0]
    map = dfr.set_index('prefix')['prediction'].to_dict()
    # print(json.dumps(map, indent=4))
    
    # print(map)
    while(s!=utt):
        pred = map.get(s, "###############################")
        # print((s, utt[len(s):], pred))
        if(utt[len(s):].startswith(pred) and pred!=""):
            s += pred
        else:
            # print(pred, "######", utt[len(s):])
            c += 1
            s += utt[len(s)]
    # print((c, len(utt)))
    return 1 - c/len(utt)
    


if __name__ == "__main__":
    input_file = "outputs/out.seen.ddc.mpc"
    lines = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            if(len(line)!=5):
                continue
            if(has_context(input_file)):
                if("<|EOU|>" not in line[0]):
                    continue
            line[0] = line[0].split("<|EOU|>")[-1]
            line[1] = line[1].split("<|EOU|>")[0]
            line[2] = line[2].split("<|EOU|>")[0]
            lines.append(line)
    blocks = make_blocks(lines)
    blocks = blocks
    thres = 300
    # print([eval_block(block, thres) for block in blocks])
    print(sum([eval_block(block, thres) for block in blocks])/len(blocks))