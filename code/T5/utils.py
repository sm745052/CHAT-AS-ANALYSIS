from copy import deepcopy
from typing import Sequence
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer

CRED = '\033[91m'
CEND = '\033[0m'


def suffix_encoder(tokenizer, text, max_length, batching=False, prev_space=True):
    if (batching):
        for i in range(len(text)):
            if (not prev_space[i]):
                text[i] = "«" + text[i]
    else:
        if (not prev_space):
            text = "«" + text
    encoded = tokenizer(
        text, padding="max_length", max_length=max_length, truncation=True,
        return_tensors='pt')
    if (batching):
        for i in range(len(encoded['input_ids'])):
            if (not prev_space[i]):
                encoded['input_ids'][i] = torch.cat(
                    (encoded['input_ids'][i][1:],
                     torch.tensor([tokenizer.pad_token_id])))
                encoded['attention_mask'][i] = torch.cat(
                    (encoded['attention_mask'][i][1:], torch.tensor([0])))
        return encoded
    if (not prev_space):
        encoded['input_ids'][0] = torch.cat(
            (encoded['input_ids'][0][1:],
             torch.tensor([tokenizer.pad_token_id])))
        encoded['attention_mask'][0] = torch.cat(
            (encoded['attention_mask'][0][1:], torch.tensor([0])))
    return encoded


def suffix_decoder(tokenizer, encoded):
    text = tokenizer.decode(
        torch.cat((torch.tensor([673]),
                   encoded),
                  dim=0),
        skip_special_tokens=True)
    text = text[1:]
    # if(len(text)==0):
    #     print("Empty text")
    #     print(encoded)
    return text


def prefix_encoder(tokenizer, text:Sequence[str], max_length, batch=False):
    if (batch):
        text = deepcopy(list(text))
        for i in range(len(text)):
            if (text[i][-1] == " "):
                text[i] = text[i][:-1] + "<tspace>"
    else:
        if (text[-1] == " "):
            text = text[:-1] + "<tspace>"
    encoded = tokenizer(
        text, padding="max_length", max_length=max_length, truncation=True,
        return_tensors='pt')
    return encoded


def merge_prefix_suffix(prefix, suffix):
    if (len(suffix) > 0 and len(prefix) > 0 and suffix[0] == " " and prefix[-1] == " "):
        return prefix[:-1] + suffix
    else:
        return prefix + suffix


class AutocompleteDataset(Dataset):
    def __init__(self, tokenizer, sentences, infer=False, tkmax_length=256,
                 context=False, debug=False):
        self.tokenizer = tokenizer
        self.context = context
        self.debug = debug
        # preprocessing
        if self.context:
            self.sentence = [
                self.preprocess_text(row) for row in sentences
                if len(self.preprocess_text(row.split('\t')[-1])) >= 2]
        else:
            self.sentence = [
                self.preprocess_text(row) for row in sentences
                if len(self.preprocess_text(row)) >= 2]
        self.max_length = tkmax_length
        self.infer = infer

    def __len__(self):
        # return 100
        return len(self.sentence)

    def __getitem__(self, idx):
        curr_sentence = self.sentence[idx]
        if self.infer == True:
            # i think both can be of same format during inference
            input_text = curr_sentence.split("\t")[0]
            target_text = curr_sentence.split("\t")[1]
            return input_text, target_text

        r = np.random.randint(1, len(curr_sentence))
        input_text = curr_sentence[:r]
        target_text = curr_sentence[r:]

        if (self.context):
            context_sentence = ' '.join(curr_sentence.split('\t')[:-1])
            breaking_sentence = curr_sentence.split('\t')[-1]
            r = np.random.randint(1, len(breaking_sentence))
            input_text = context_sentence + breaking_sentence[:r]
            target_text = breaking_sentence[r:]

        # print("input = ", [input_text])
        # print("target = ", [target_text])
        if self.debug:
            print(CRED + input_text + CEND + target_text)

        input_encoded = prefix_encoder(
            self.tokenizer, input_text, max_length=self.max_length)
        target_encoded = suffix_encoder(
            self.tokenizer, target_text, max_length=self.max_length,
            prev_space=(input_text[-1] == " "))
        # print("input = ", [input_text])
        # print("target = ", [target_text])
        # print("target decoded = ", [suffix_decoder(self.tokenizer, target_encoded['input_ids'][0])])
        return input_encoded, target_encoded

    def preprocess_text(self, text):
        text = text.strip().lower()
        text = text.replace("<eou>", "<|EOU|>")
        return text


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", truncation_side='left', model_max_length=256)
    sentences = ['Hi myname is sandeep<eou>\tOh hi! Sandeep']
    dataset = AutocompleteDataset(tokenizer, sentences, debug=True, context=True)
    for i in range(100):
        dataset.__getitem__(0)
