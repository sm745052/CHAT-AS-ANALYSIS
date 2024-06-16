import pandas as pd


def space_correction(x):
    x = x.replace(" ,", ",")
    x = x.replace(" .", ".")
    x = x.replace(" ?", "?")
    x = x.replace(" !", "!")
    return x



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
    print(map)
    # print(map)
    unique_utterances = set() # set of unique utterances
    
    tes = 0
    tot_chars = 0
    next_prefix = ""
    
    with open (raw_tst_file, "r") as f:
        for line in f:
            prefix, suffix = preprocess(line, model)
            if len(unique_utterances)==0:
                unique_utterances.add(prefix + suffix) # add the new prefix and suffix to the set
                tot_chars = len(prefix+suffix) # update the tot_chars
                tes = 0 
            elif (prefix + suffix) not in unique_utterances:
                saved_ks.append(tes/tot_chars) # save the tes value for the previous utterance
                unique_utterances.add(prefix + suffix) # add the new prefix and suffix to the set
                tot_chars = len(prefix+suffix) # update the tot_chars
                tes = 0 # reset the tes
    
            # if prefix is in map then check if gt starts with prediction
            if prefix in map:
                # print(prefix)
                if suffix.startswith(map[prefix]) and not next_prefix.startswith(prefix):
                    tes += len(map[prefix])
                    # print(prefix, tes, "\n")
                    next_prefix = prefix + map[prefix] # since we accepted the whole prediction and hence we start with the prefix after this
    
    saved_ks.append(tes/tot_chars)
     
    # print(saved_ks)
    tes_metric = sum(saved_ks)/len(saved_ks)
    print("TES Metric: ", tes_metric)

    return tes_metric