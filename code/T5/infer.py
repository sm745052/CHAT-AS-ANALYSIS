import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import argparse
import os
from accelerate import Accelerator
import tqdm
import numpy as np
import sys
sys.path.append('./code')
from T5.utils import AutocompleteDataset, merge_prefix_suffix, prefix_encoder, suffix_decoder, suffix_encoder


accelerator = Accelerator()

device = accelerator.device
print("PROCESS STARTED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--bs',  type=int, default=4)
    parser.add_argument('--tkmax_length', type=int, default=256)
    parser.add_argument('--mdmax_length', type=int, default=256)
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()

    print("Using device:", device)

    # load tokenizer
    print("Loading tokenizer and model...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side='left', max_length=args.tkmax_length)
    tokenizer.add_tokens('<tspace>')
    if(args.context):
        tokenizer.add_tokens('<|EOU|>')
    model.resize_token_embeddings(len(tokenizer))

    if(args.ckpt is not None):
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Checkpoint loaded - ", args.ckpt)
    else:
        print("No checkpoint loaded, using vanilla pretrained model")
    model.eval()

    # load data
    print("Loading data...")
    with open(args.inp, "r") as f:
        data = f.read()
    dataset = data.split("\n")[:-1]
    infer_data = pd.DataFrame(dataset)
    sentences = infer_data.values.flatten().tolist()
    print("Preparing data...")
    dataset = AutocompleteDataset(tokenizer, sentences, tkmax_length=args.tkmax_length, infer=True, context=args.context)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False)
    print("Data prepared")

    model, data_loader = accelerator.prepare(model, data_loader)

    print("Inferencing...")
    with open(args.out, "w") as f:
        for batch in tqdm.tqdm(data_loader):
            inputs, targets = batch
            inputs = list(inputs)
            targets = list(targets)
            # Prepare data
            encoding = prefix_encoder(tokenizer, inputs, max_length=args.tkmax_length, batch=True)
            encoding = encoding.to(device)
            # generate outputs with attention masks
            if(hasattr(model, "module")):
                generated_outputs = model.module.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, num_beams=3, max_new_tokens=args.mdmax_length, early_stopping = True, return_dict_in_generate=True, output_scores=True)
            else:
                generated_outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, num_beams=3, max_new_tokens=args.mdmax_length, early_stopping = True, return_dict_in_generate=True, output_scores=True)

            #print the generated sequences
            # print("Generated sequences:")
            # print(tokenizer.batch_decode(generated_outputs.sequences, skip_special_tokens=True))
            # exit(0)

            gen_sequences = generated_outputs.sequences[:, 1:] # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for # encoder-decoder models, like BART or T5.

            # let's stack the logits generated at each step to a tensor and transform
            # logits to probs
            probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1) # -> shape [3, 15, vocab_size]

            # now we need to collect the probability of the generated token
            # we need to add a dummy dim in the end to make gather work
            # print(probs.shape)
            # print(gen_sequences.shape)
            try:
                gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            except:
                print("Exception occured while calculating probabilities")
                print(gen_sequences)
                print(probs)
                print(tokenizer.batch_decode(gen_sequences, skip_special_tokens=True))
                exit(0)

            # get the average negative log likelihood across generated tokens for each sequence that are not pad tokens

            mask = gen_sequences != tokenizer.pad_token_id
            mask = mask.type(torch.FloatTensor).to(device)

            nll = -torch.log(gen_probs) * mask
            nll = nll.sum(1)
            subword_lens = mask.sum(1)
            gen_sequences = gen_sequences.cpu()
            for i in range(len(inputs)):
                prefix = tokenizer.decode(encoding.input_ids[i], skip_special_tokens=True)
                prefix = prefix.replace("<tspace>", " ")
                gt = targets[i]
                pred = suffix_decoder(tokenizer, gen_sequences[i])
                total_sentence = merge_prefix_suffix(prefix, pred)
                pred = total_sentence[len(prefix):]
                confidence = str(nll[i].item())
                subword_len = str(subword_lens[i].item())
                # print(total_sentence)
                print([prefix, gt, pred, confidence, str(subword_len)])
                f.write("\t".join([prefix, gt, pred, confidence, str(subword_len)]) + "\n")