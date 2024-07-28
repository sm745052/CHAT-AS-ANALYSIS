from termcolor import colored
import numpy as np
from transformers import AutoTokenizer
from types import SimpleNamespace
from typing import List, Callable, Any
import thres_ckecks
from tqdm import tqdm
import multiprocessing


def predict_thres_check_function(outfile: str) -> Callable[[Any], bool]:
    outfile = outfile.lower()
    if "qb" in outfile:
        return thres_ckecks.thres_checkQB
    elif "gpt2" in outfile or "t5" in outfile:
        return thres_ckecks.thres_checkGPT2
    elif "mpc" in outfile:
        return thres_ckecks.thres_checkMPC
    elif "ngame" in outfile:
        return thres_ckecks.thres_checkNGAME
    elif "gpt4" in outfile:
        return thres_ckecks.thres_checkGPT4
    else:
        raise NotImplementedError


def predict_model_name(outfile: str) -> str:
    outfile = outfile.lower()
    if "t5" in outfile:
        if "small" in outfile:
            return "t5-small"
        elif "mini" in outfile:
            return "t5-mini"
        else:
            return "t5-base"
    elif "qb" in outfile:
        return "qb"
    elif "mpc" in outfile:
        return "mpc"
    return ""


def predict_test_formatted_file(outfile: str) -> str:
    outfile = outfile.lower()
    res = "data/"
    if "cddc" in outfile:
        res += "cDDC/"
    elif "ddc" in outfile:
        res += "DDC/"
    elif "cdstc7" in outfile:
        res += "cDSTC7/"
    elif "dstc7" in outfile:
        res += "DSTC7/"

    if "unseen" in outfile:
        res += "unseen/"
    elif "seen" in outfile:
        res += "seen/"
    elif "all" in outfile:
        res += "all/"

    return res + "test_formatted.txt"


def is_empty(text):
    return len(text) == 0 or text == ""


def tes_metric(char_saved, total_query_len):
    return char_saved / total_query_len


def has_context(outfile: str) -> str:
    outfile = outfile.lower()
    return "cddc" in outfile or "cdstc7" in outfile


def get_args(outfile: str) -> SimpleNamespace:
    args = SimpleNamespace()
    args.max_length = 256
    args.model_name = predict_model_name(outfile)
    args.test_formatted_file = predict_test_formatted_file(outfile)
    args.has_context = has_context(outfile)
    return args


class TES:
    def __init__(self, outfile):
        self.args = get_args(outfile)
        self.tokenizer = self.get_tokenizer()
        with open(outfile, "r") as f:
            self.outlines = f.readlines()
        self.model_pred = {}
        self.thres_check = predict_thres_check_function(outfile)

    def set_model_pred(self, thres):
        self.model_pred = {}
        print(colored(f"Setting model predictions for {thres}", "green"))
        for i in tqdm(self.outlines):
            i = i.split("\t")
            if self.thres_check(*i, thres):
                self.model_pred[i[0]] = i[2]

    def get_tokenizer(self):
        if "t5" in self.args.model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name, truncation_side="left"
            )
            tokenizer.add_tokens("<tspace>")
            tokenizer.add_tokens("<|EOU|>")
            return tokenizer
        elif "qb" in self.args.model_name:
            return None
        elif "mpc" in self.args.model_name:
            return None
        raise NotImplementedError

    def format_query(self, text):
        if "t5" in self.args.model_name:
            flag = bool(len(text) and text[-1] == " ")
            if flag:
                text = text[:-1] + "<tspace>"
            return self.tokenizer.decode(
                self.tokenizer.encode(
                    text, truncation=True, max_length=self.args.max_length
                ),
                skip_special_tokens=True,
            ).replace("<tspace>", " ")
        if "qb" in self.args.model_name:
            return text
        if "mpc" in self.args.model_name:
            return text
        return text

    def model_preprocess_mock(self, text):
        if "t5" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            return text
        if "qb" in self.args.model_name:
            text = text.strip().lower()
            return text
        if "mpc" in self.args.model_name:
            text = text.strip().lower()
            return text
        raise NotImplementedError

    def rformat(self, x):
        if "t5" in self.args.model_name:
            x = x.replace(",", " ,")
            x = x.replace(".", " .")
            x = x.replace("?", " ?")
            x = x.replace("!", " !")
            x = x.replace("’", " ’")
            return x
        if "qb" in self.args.model_name:
            return x
        if "mpc" in self.args.model_name:
            return x
        raise NotImplementedError

    def get_unique_utterances(self) -> List[str]:
        with open(self.args.test_formatted_file, "r") as f:
            lines = f.readlines()
        uts = [
            self.model_preprocess_mock(f"{i}{j}")
            for i, j in [i.split("\t") for i in lines]
        ]
        return list(set(uts))

    def custom_pruning(self, uts: List[str]) -> List[str]:
        if self.args.has_context:
            uts = list(filter(lambda x: "<|EOU|>" in x, uts))
            return uts
        return uts

    def calculate_metric(self, input_query, context, verbose=False):
        saved_chars = 0
        prefix = ""

        total_query = context + input_query
        found = 0
        tried = 0
        matched = 0
        i = len(context) + 1

        while i < len(total_query):
            tried += 1
            prefix = self.format_query(total_query[:i])
            gt_suffix = total_query[i:]
            if verbose:
                print(f"{prefix}{colored('<-->', 'red')}{gt_suffix}")
            if prefix in self.model_pred and not is_empty(self.model_pred):
                temp_sugg = self.rformat(self.model_pred[prefix])
                found += 1
                if verbose:
                    print(f"{colored('found', 'green')}: {colored(temp_sugg, 'blue')}")
                if gt_suffix.startswith(temp_sugg):
                    matched += 1
                    if verbose:
                        print(colored("MATCHED", "cyan"))
                    saved_chars += len(temp_sugg)
                    i += len(temp_sugg)
                else:
                    i += 1
            else:
                if verbose:
                    print(colored("not found", "red"))
                i += 1
        final_score = tes_metric(saved_chars, len(input_query))
        if verbose:
            print(colored(f"TES: {final_score}", "green"))
        return final_score, found / tried, matched

    def prepare(self, idx):
        if "t5" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, context
        if "qb" in self.args.model_name:
            return self.uts[idx], ""
        if "mpc" in self.args.model_name:
            return self.uts[idx], ""
        raise NotImplementedError

    def e2e(self, idx, verbose=False):
        tes_, found, length = self.calculate_metric(*self.prepare(idx), verbose=verbose)
        return [tes_, found, length]

    def run(self, thres, multi=False, cores=8, subset=None, verbose=False):
        P = multiprocessing.Pool(cores)
        self.set_model_pred(thres)
        self.uts = self.get_unique_utterances()
        self.uts = self.custom_pruning(self.uts)
        idxs = (
            np.random.choice(len(self.uts), subset) if subset else range(len(self.uts))
        )
        if multi:
            res = P.map(self.e2e, idxs)
        else:
            res = [self.e2e(i, verbose) for i in tqdm(idxs)]
        P.close()
        return np.array(res).reshape(-1, 3)


if __name__ == "__main__":
    outfile = "outputs/out.unseen.ddc.t5"
    tes = TES(outfile)
    res = tes.run(10000, multi=False, verbose=False, cores=30, subset=None)
    print(colored("TES", "green"), " : ", np.mean(res[:, 0]))
    # print(colored("Found percent", "green"), " : ", np.mean(res[:, 1]))
    print(colored("average number of matches", "green"), " : ", np.mean(res[:, 2]))
