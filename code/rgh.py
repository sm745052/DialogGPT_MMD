import os

# os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import wandb
import torch
from datasets import Dataset
import argparse
from typing import List, Dict, Union
from dataclasses import dataclass
from dataset import (
    IMAGE_TOKEN,
    MARKER_TOKEN,
    CaptionDataset,
    DIALOGCC_DATA_PATH,
)
from chatas.code.utils.dataset import (
    Dialog,
    Utterance,
    DialogData,
)
from collections import defaultdict
import string
import logging

import time
from tqdm import tqdm


MODEL = "microsoft/dialogpt-medium"
CKPT = "ckpt/dialogcc/checkpoint-1500"

OUTPUT_FOLDER = "."


BATCH_SIZE = 32

TOKENIZER_MAX_LENGTH = 256

class Trie:
    def __init__(self, tokenizer):
        self.trie = defaultdict(lambda: set())
        self.seen = set()
        self.tokenizer = tokenizer

    def add_word(self, word):
        word = " " + word
        if word not in self.seen:
            self.seen.add(word)
            tokens = self.tokenizer.tokenize(word)
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            assert len(tokens) == len(token_ids)  # for self check
            if (
                len(word) != sum([len(token) for token in tokens])
                or word[1:] != "".join(tokens)[1:]
            ):
                logging.warning(
                    f"COULDNOT BE USED DUE TO UNUSUAL TOKENIZATION  Word: {word}, Tokens: {tokens}"
                )
            token_len = [len(token) for token in tokens]
            prefix_sum = [token_len[0]]
            for i in range(1, len(token_len)):
                prefix_sum.append(prefix_sum[i - 1] + token_len[i])
            for i in range(1, len(word) + 1):
                ctr = 0
                for j in range(len(prefix_sum)):
                    if i <= prefix_sum[j]:
                        ctr = j
                        break
                self.trie[word[:i]].add(tuple(token_ids[: ctr + 1]))

    def __getitem__(self, key):
        if key not in self.trie:
            token_ids = self.tokenizer.encode(key, add_special_tokens=False)
            self.trie[key] = set([tuple(token_ids)])
        return self.trie[key]

    def __len__(self):
        return len(self.trie)


class SubwordTrie:
    def __init__(self):
        self.trie = defaultdict(lambda: list())

    def add_token_id_sequence(self, token_id_sequence):
        for i in range(0, len(token_id_sequence)):
            # print(f"Adding {token_id_sequence[:i]} -> {token_id_sequence[i]}")
            self.trie[tuple(token_id_sequence[:i])].append(
                token_id_sequence[i]
            )

    def __getitem__(self, key):
        # print(f"Querying {key=}")
        return self.trie[key]


def get_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(
        model, truncation=True, truncation_side="left", padding="max_length", padding_side="left"
    )
    return tokenizer




def preprocess_line(line):
    line = line.lower()
    # remove punctuation
    for p in string.punctuation:
        line = line.replace(p, " ")
    line = line.replace("“", " ")
    line = line.strip()
    return line


def get_words_from_line(line):
    return line.split()


def make_trie(dialogdata, tokenizer):
    trie = Trie(tokenizer)
    for dialog,_ in dialogdata.raw_data:
        for utterance in dialog:
            line = utterance.text
            line = preprocess_line(line)
            words = get_words_from_line(line)
            for word in words:
                trie.add_word(word)
    return trie


def make_prefix_allowed_tokens_fn(
    config_by_batch_id: Dict[int, Dict[str, Union[str, int]]],
    full_vocab: list,
):

    def prefix_allowed_tokens_fn(batch_id: int, sent: torch.LongTensor):
        subword_trie = config_by_batch_id[batch_id]["subword_trie"]
        sent_orig_len = config_by_batch_id[batch_id]["sent_orig_len"]
        # print(f"{sent_orig_len=}")
        # print(f"{len(sent)=}")
        added_token_ids = sent[sent_orig_len:].tolist()
        added_tokens = [
            trie.tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in added_token_ids
        ]
        # print(f"{added_tokens=}")
        probable_completions = subword_trie[tuple(added_token_ids)]
        probable_completions = list(set(probable_completions))
        probable_completion_tokens = [
            trie.tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in probable_completions
        ]
        # print(f"{probable_completion_tokens=}")
        if len(probable_completions) == 0:
            return full_vocab
        return probable_completions

    return prefix_allowed_tokens_fn





def encode(tokenizer, text):
    utterances = text.replace("<eou>", tokenizer.eos_token)
    enc = tokenizer(
        utterances,
        max_length=TOKENIZER_MAX_LENGTH,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return enc


def encode_batch(tokenizer, texts):
    utterances = [text.replace("<eou>", tokenizer.eos_token) for text in texts]
    enc = tokenizer(
        utterances,
        max_length=TOKENIZER_MAX_LENGTH,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding_side="left",
        padding = "max_length",
    )
    return enc


def get_lines(file):
    with open(file, "r") as f:
        lines_ = f.readlines()
    lines_ = lines_[len(lines_) // 2:]
    lines = [line.split("\t")[0] for line in lines_]
    preds = [
        line.split("\t")[1].strip("\n") if len(line.split("\t")) > 1 else ""
        for line in lines_
    ]
    for i in range(len(lines)):
        last_uttr = lines[i].split("<eou>")[-1]
        if lines[i][: -len(last_uttr)].endswith("<eou>"):
            lines[i] = lines[i][: -len(last_uttr)] + " " + last_uttr
    # print(f"{lines[0]=}")
    return lines, preds


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(CKPT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    # model.resize_token_embeddings(len(tokenizer))
    assert model.config.vocab_size == len(tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    full_vocab = list(range(len(tokenizer)))
    model.config.pad_token_id = tokenizer.eos_token_id


    train_dataset = CaptionDataset(tokenizer, f"{DIALOGCC_DATA_PATH}test.csv", test = False)
    trie = make_trie(train_dataset, tokenizer)

    tokens = (trie[" one."])
    print(tokens)
    for token in tokens:
        print(tokenizer.decode(token))