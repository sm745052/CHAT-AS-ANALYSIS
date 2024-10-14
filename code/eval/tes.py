import cProfile
from collections import defaultdict
from termcolor import colored
import numpy as np
from transformers import AutoTokenizer
from types import SimpleNamespace
from typing import List, Callable, Any, Tuple, Union, Sequence
import thres_ckecks
from tqdm import tqdm
import argparse
import os
import sys

sys.path.append("code")

from T5 import utils as T5utils

os.environ["TOKENIZERS_PARALLELISM"] = "true"

NO_CUSTOM_PRUNING_FOR_CONTEXT_MODELS = ['T5', '']


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
    elif "mistral" in outfile:
        return thres_ckecks.thres_checkGPT2
    elif "phi" in outfile:
        return thres_ckecks.thres_checkGPT2
    elif "renee" in outfile:
        return thres_ckecks.thres_checkMPC
    else:
        raise NotImplementedError


def predict_model_name(outfile: str) -> str:
    outfile = outfile.lower()
    res = ""
    if "t5" in outfile:
        res = "t5"
    elif "qb" in outfile:
        res = "qb"
    elif "mpc" in outfile:
        res = "mpc"
    elif "mistral" in outfile:
        res = "mistral"
    elif "gpt2" in outfile:
        res = "gpt2"
    elif "phi" in outfile:
        res = "phi"        
    elif "gpt4" in outfile:
        res = "gpt4"
    elif "renee" in outfile:
        res = "renee"

    if ".c2" in outfile:
        res += ".c2"
    elif ".c4" in outfile:
        res += ".c4"
    elif ".cinf" in outfile:
        res += ".cinf"

    return res


def predict_test_formatted_file(outfile: str) -> str:
    outfile = outfile.lower()
    res = "data/"
    if "cddc" in outfile:
        res += "NcDDC/"
    elif "ddc" in outfile:
        res += "DDC/"
    elif "cdstc7" in outfile:
        res += "cDSTC7/"
    elif "dstc7" in outfile:
        res += "DSTC7/"
    if 'unseen' in outfile:
        res += "unseen/"
    elif 'seen' in outfile:
        res += "seen/"
    elif 'all' in outfile:
        res += "all/"
    return res + "test_formatted.txt"


def is_empty(text):
    return len(text) == 0 or text == ""


def tes_metric(char_saved, total_query_len):
    if total_query_len == 0:
        return np.nan
    return char_saved / total_query_len


def has_context(outfile: str) -> str:
    outfile = outfile.lower()
    return "cddc" in outfile or "cdstc7" in outfile


def get_args(outfile: str, truncate: int, max_context: int) -> SimpleNamespace:
    args = SimpleNamespace()
    args.outfile = outfile
    args.max_length = 256
    args.model_name = predict_model_name(outfile)
    args.test_formatted_file = predict_test_formatted_file(outfile)
    args.has_context = has_context(outfile)
    args.truncate = truncate
    args.max_context = max_context
    return args


def get_prefixes(sent: str, context: str) -> List[str]:
    return [context + sent[:i] for i in range(1, len(sent) + 1)]


def space_correction(prefix: str) -> str:
    prefix = prefix.replace(" .", ".")
    prefix = prefix.replace(" ?", "?")
    prefix = prefix.replace(" !", "!")
    prefix = prefix.replace(" ,", ",")
    prefix = prefix.replace(" ’", "’")
    return prefix


def truncate(x: str, n: int) -> str:
    x = x.split(" ")
    x = x[:n]
    return " ".join(x)


class TES:
    def __init__(self, outfile: str, truncate: int = None, max_context: int = None):
        self.args = get_args(outfile, truncate, max_context)
        self.tokenizer = self.get_tokenizer()
        with open(outfile, "r") as f:
            self.outlines = f.readlines()
        self.model_pred = {}
        self.thres_check = predict_thres_check_function(outfile)
        self.uts = self.get_unique_utterances()
        self.uts = self.custom_pruning(self.uts)
        print(colored("Preprocessing prefixes", "green"))
        self.prefix_transform = self.create_prefix_transform()
        # print a subset of the prefixes kewy value in the dict
        # print({k: self.prefix_transform[k] for k in list(self.prefix_transform)[:5]})
        print(colored("Preprocessing prefixes done", "green"))

    def create_prefix_transform(self):
        all_prefixes = [
            prefix
            for idx in range(len(self.uts))
            for prefix in get_prefixes(*self.prepare(idx))
        ]

        # Use the batched format_prefix method
        formatted_prefixes = self.format_prefix(all_prefixes, batch_size=2048 * 2)

        return defaultdict(
            str,
            {prefix: formatted_prefixes[i] for i, prefix in enumerate(all_prefixes)},
        )

    def set_model_pred(self, thres: Union[float, str]):
        self.model_pred = {}
        # print(colored(f"Loading model predictions for threshold={thres}", "green"))
        for i in self.outlines:
            i = i.split("\t")
            if len(i) != 5:
                continue
            if self.thres_check(*i, thres):
                self.model_pred[i[0]] = truncate(i[2], self.args.truncate)
        # print(colored(f"Model predictions loaded {len(self.model_pred)}", "green"))

    def get_tokenizer(self):
        if "t5" in self.args.model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                't5-base', truncation_side="left"
            )
            tokenizer.add_tokens("<tspace>")
            tokenizer.add_tokens("<|EOU|>")
            return tokenizer
        if "gpt2" in self.args.model_name:
            return None
        elif "qb" in self.args.model_name:
            return None
        elif "mpc" in self.args.model_name:
            return None
        elif "mistral" in self.args.model_name:
            return None
        elif "phi" in self.args.model_name:
            return None
        elif "gpt4" in self.args.model_name:
            return None
        elif "renee" in self.args.model_name:
            return None
        raise NotImplementedError

    def format_prefix(self, all_prefixes: Sequence[str], batch_size=64) -> List[str]:
        formatted_prefixes = []
        for i in tqdm(range(0, len(all_prefixes), batch_size)):
            formatted_prefixes += self.format_prefix_batch(
                all_prefixes[i : i + batch_size]
            )
        return formatted_prefixes

    def format_prefix_batch(self, all_prefixes: Sequence[str]) -> List[str]:
        if "t5" in self.args.model_name:
            return [
                self.tokenizer.decode(enc, skip_special_tokens=True).replace(
                    "<tspace>", " "
                )
                for enc in T5utils.prefix_encoder(
                    self.tokenizer,
                    all_prefixes,
                    max_length=self.args.max_length,
                    batch=True,
                ).input_ids
            ]
        if "qb" in self.args.model_name:
            return all_prefixes
        if "mpc" in self.args.model_name:
            return all_prefixes
        if "gpt2" in self.args.model_name:
            return all_prefixes
        if "mistral" in self.args.model_name:
            return all_prefixes
        if "phi" in self.args.model_name:
            return all_prefixes
        if "gpt4" in self.args.model_name:
            return all_prefixes
        if "renee" in self.args.model_name:
            return all_prefixes
        return all_prefixes

    def model_preprocess_mock(self, text, max_context = None):
        if "t5" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            if max_context:
                context = "<|EOU|>".join(text.split("<|EOU|>")[:-1])
                utterance = text.split("<|EOU|>")[-1]
                new_context = "<|EOU|>".join(context.split("<|EOU|>")[-max_context:])
                text = new_context + "<|EOU|>" + utterance
            return text
        if "gpt2" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            if max_context:
                context = "<|EOU|>".join(text.split("<|EOU|>")[:-1])
                utterance = text.split("<|EOU|>")[-1]
                new_context = "<|EOU|>".join(context.split("<|EOU|>")[-max_context:])
                text = new_context + "<|EOU|>" + utterance
            return text
        if "qb" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            return text
        if "mpc" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            return text
        if "mistral" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            return text
        if "phi" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            return text
        if "gpt4" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            return text
        if "renee" in self.args.model_name:
            text = text.strip().lower()
            text = text.replace("<eou>", "<|EOU|>")
            return text
        raise NotImplementedError

    def rformat(self, x):
        # print(self.args.model_name)
        if "ddc" in self.args.outfile:
            if "t5" in self.args.model_name:
                x = x.replace(",", " ,")
                x = x.replace(".", " .")
                x = x.replace("?", " ?")
                x = x.replace("!", " !")
                x = x.replace("’", " ’")
                return x
            if "gpt2" in self.args.model_name:
                x = x.replace(",", " ,")
                x = x.replace(".", " .")
                x = x.replace("?", " ?")
                x = x.replace("!", " !")
                x = x.replace("’", " ’")
                x = x.replace("<|EOU|>", "")
                return x
            if "qb" in self.args.model_name:
                return x
            if "mpc" in self.args.model_name:
                return x
            if "mistral" in self.args.model_name:
                return x
            if "phi" in self.args.model_name:
                return x
            if "gpt4" in self.args.model_name:
                return x
            if "renee" in self.args.model_name:
                return x
            raise NotImplementedError
        if "gpt2" in self.args.model_name:
            x = x.replace("<|EOU|>", "")
            return x
        return x

    def get_unique_utterances(self) -> List[str]:
        with open(self.args.test_formatted_file, "r") as f:
            lines = f.readlines()
        uts = [
            self.model_preprocess_mock(f"{i}{j}", self.args.max_context)
            for i, j in [i.split("\t") for i in lines]
        ]
        return list(set(uts))

    def custom_pruning(self, uts: List[str]) -> List[str]:
        if self.args.has_context:
            if True not in (
                [
                    model in self.args.model_name
                    for model in NO_CUSTOM_PRUNING_FOR_CONTEXT_MODELS
                ]
            ):
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
        matched_ḻengths = []
        i = len(context) + 1

        while i < len(total_query):
            tried += 1
            prefix = self.prefix_transform[total_query[:i]]
            gt_suffix = total_query[i:]
            if verbose:
                print(f"{total_query[:i]}{colored('<-->', 'red')}{gt_suffix}")
            if prefix in self.model_pred and not is_empty(
                self.rformat(self.model_pred[prefix])
            ):
                temp_sugg = self.rformat(self.model_pred[prefix])
                found += 1
                if is_empty(temp_sugg):
                    continue
                if verbose:
                    print(f"{colored('found', 'green')}: {colored(temp_sugg, 'blue')}")
                if gt_suffix.startswith(temp_sugg):
                    matched += 1
                    if verbose:
                        print(colored("MATCHED", "cyan"))
                    saved_chars += len(temp_sugg)
                    matched_ḻengths.append(len(temp_sugg))
                    i += len(temp_sugg)
                else:
                    i += 1
            else:
                if verbose:
                    print(
                        f'{colored("not found", "red")}: checked "{colored(self.prefix_transform[total_query[:i]], "blue")}"'
                    )
                i += 1
        final_score = tes_metric(saved_chars, len(input_query))
        if verbose:
            print(colored(f"TES: {final_score}", "green"))
        return (
            final_score,
            found / tried if tried else np.nan,
            matched,
            (np.mean(matched_ḻengths) if matched_ḻengths else np.nan),
        )

    def prepare(self, idx: int) -> Tuple[str, str]:
        if "t5" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, context
        if "gpt2" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, context
        if "qb" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, ""
        if "mpc" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, ""
        if "mistral" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, ""
        if "phi" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, ""
        if "gpt4" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, ""
        if "renee" in self.args.model_name:
            ch = self.uts[idx]
            context = "<|EOU|>".join(ch.split("<|EOU|>")[:-1])
            if context:
                context += "<|EOU|>"
            sent = ch.split("<|EOU|>")[-1]
            return sent, ""
        raise NotImplementedError

    def e2e(self, idx, verbose=False):
        tes_, found, matched, ghost_length = self.calculate_metric(
            *self.prepare(idx), verbose=verbose
        )
        return [tes_, found, matched, ghost_length]

    def run(self, thres: Union[float, str], subset=None, verbose=False):
        self.set_model_pred(thres)
        idxs = (
            np.random.choice(len(self.uts), subset) if subset else range(len(self.uts))
        )
        res = [self.e2e(i, verbose) for i in idxs]
        return np.array(res).reshape(-1, 4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--outfile", type=str, default="outputs/out.seen.ddc.phi.finetune.word3"
    )
    argparser.add_argument("--subset", type=int, default=None)
    argparser.add_argument("--max_context", type=int, default=None)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()
    for k, v in vars(args).items():
        print(colored(k, "green"), ":", v, end="  ")
    print()

    tes = TES(args.outfile, max_context = args.max_context)
    res = tes.run(
        10000,
        verbose=args.verbose,
        subset=args.subset,
    )
    # print(res)
    print(colored("outfile", "green"), " : ", args.outfile)
    print(colored("TES", "green"), " : ", np.nanmean(res[:, 0]))
    print(colored("Found percent", "green"), " : ", np.nanmean(res[:, 1]))
    print(colored("average number of matches", "green"), " : ", np.nanmean(res[:, 2]))
    print(colored("average matched length", "green"), " : ", np.nanmean(res[:, 3]))
