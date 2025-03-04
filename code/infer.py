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
import pandas as pd
import time
from tqdm import tqdm


logging.basicConfig(level="WARNING")

MODEL = "microsoft/dialogpt-medium"

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
    logging.info(f"{utterances=}")
    enc = tokenizer(
        utterances,
        max_length=TOKENIZER_MAX_LENGTH,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding_side="left",
        padding = "max_length",
        # truncation_side="left",
    )
    dec = []
    for i in range(len(utterances)):
        dec.append(tokenizer.decode(enc["input_ids"][i], skip_special_tokens=False))
    logging.info(f"{dec=}")
    return enc



def predict_normal(inp: str, tokenizer, model):
    enc = encode(tokenizer, inp)
    output = model.generate(
        enc,
        top_k=10,
        top_p=0.95,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        do_sample = True,
    )
    completion = tokenizer.decode(output[0], skip_special_tokens=False)
    input_as_seen = tokenizer.decode(enc[0], skip_special_tokens=False)
    return completion[len(input_as_seen) :]



def predict_adv(inp: str, trie: Trie, tokenizer, model, full_vocab):
    subword_trie = SubwordTrie()
    if inp[-1] == " ":
        main_sent = inp[:-1]
        last_word = ""
    else:
        main_sent = " ".join(inp.split()[:-1])
        last_word = inp.split()[-1]
    # print(f"{main_sent=}")
    # print(f"{last_word=}")
    if last_word == "":
        probable_completions = []
    else:
        probable_completions = trie[" " + last_word]
        # probable_completions_tokens = [
        #     [
        #         tokenizer.decode([token_id], skip_special_tokens=False)
        #         for token_id in token_sequence
        #     ]
        #     for token_sequence in probable_completions
        # ]
        # print(f"{probable_completions_tokens=}")
    for completion in probable_completions:
        subword_trie.add_token_id_sequence(completion)
    # print("making config")
    enc = encode(tokenizer, main_sent)
    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
        {
            0: {
                "subword_trie": subword_trie,
                "sent_orig_len": len(enc["input_ids"][0]),
            }
        },
        full_vocab,
    )
    output = model.generate(
        enc,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        top_k=10,
        top_p=0.95,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        do_sample = True,
    )
    completion = tokenizer.decode(output[0], skip_special_tokens=False)
    input_as_seen = (
        tokenizer.decode(enc[0], skip_special_tokens=False) + " " + last_word
    )
    return completion[len(input_as_seen) :]


def predict(inp: str, trie, tokenizer, model, full_vocab):
    if len(inp.split(" ")) > 1:
        res = predict_adv(inp, trie, tokenizer, model, full_vocab)
    else:
        res = predict_normal(inp, tokenizer, model)
    res = res.replace(tokenizer.eos_token, "")
    return res


def predict_normal_batch(inp: List[str], tokenizer, model):
    if not inp:
        return []
    encs = encode_batch(tokenizer, inp)
    encs["input_ids"] = encs["input_ids"].to("cuda")
    encs["attention_mask"] = encs["attention_mask"].to("cuda")
    output = model.generate(
        **encs,
        top_k=10,
        top_p=0.95,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        do_sample = True,
    )
    completions = tokenizer.batch_decode(output, skip_special_tokens=False)
    input_as_seen = tokenizer.batch_decode(encs["input_ids"], skip_special_tokens=False)
    logging.info(f"FROM NORMAL {completions=}")
    logging.info(f"FROM NORMAL {input_as_seen=}")
    return [
        completion[len(input_as_seen[i]) :].replace(tokenizer.eos_token, "")
        for i, completion in enumerate(completions)
    ]


def get_input_as_seen(enc, last_word, tokenizer):
    if len(enc) == 0:
        return last_word
    if last_word == "" and (enc[-1] == tokenizer.eos_token_id or enc[-1] == tokenizer.pad_token_id or tokenizer.decode(enc[-1], skip_special_tokens=False) == IMAGE_TOKEN):
        return tokenizer.decode(enc, skip_special_tokens=False)
    return tokenizer.decode(enc, skip_special_tokens=False) + " " + last_word


def predict_adv_batch(
    inp: List[str], trie: Trie, tokenizer, model, full_vocab
):
    if not inp:
        return []
    subword_tries = [SubwordTrie() for _ in range(len(inp))]
    main_sents = []
    last_words = []
    for i in inp:
        if i[-1] == " ": 
            main_sent = i[:-1]
            last_word = ""
        elif i.endswith(IMAGE_TOKEN):
            main_sent = i
            last_word = ""
        elif i.endswith(tokenizer.eos_token):
            main_sent = i
            last_word = ""
        else:
            last_incomplete_word = i.split(" ")[-1].split(IMAGE_TOKEN)[-1].split(tokenizer.eos_token)[-1]
            main_sent = i[:-len(last_incomplete_word)].rstrip()
            last_word = last_incomplete_word
        # print(f"{main_sent=}")
        # print(f"{last_word=}")
        logging.info(f"{main_sent=}")
        logging.info(f"{last_word=}")
        main_sents.append(main_sent)
        last_words.append(last_word)
    probable_completions = [
        trie[" " + last_word] if last_word != "" else []
        for last_word in last_words
    ]
    logging.info(f"{probable_completions=}")
    for i, probable_completion in enumerate(probable_completions):
        for completion in probable_completion:
            subword_tries[i].add_token_id_sequence(completion)
    encs = encode_batch(tokenizer, main_sents)
    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
        {
            i: {
                "subword_trie": subword_tries[i],
                "sent_orig_len": encs["input_ids"][i].shape[0],
            }
            for i in range(len(inp))
        },
        full_vocab,
    )
    # print(encs)
    encs["input_ids"] = encs["input_ids"].to("cuda")
    encs["attention_mask"] = encs["attention_mask"].to("cuda")
    outputs = model.generate(
        **encs,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        top_k=10,
        top_p=0.95,
        max_new_tokens=20,
        do_sample = True,
        pad_token_id=tokenizer.eos_token_id,
    )
    completions = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    input_as_seen = [
        get_input_as_seen(enc, last_word, tokenizer)
        for enc, last_word in zip(encs["input_ids"], last_words)
    ]
    # print(f"FROM ADV {completions=}")
    # print(f"FROM ADV {input_as_seen=}")
    logging.info(f"FROM ADV {completions=}")
    logging.info(f"FROM ADV {input_as_seen=}")
    return [
        completion[len(input_as_seen[i]) :].replace(tokenizer.eos_token, "")
        for i, completion in enumerate(completions)
    ]


def predict_batch(inp: List[str], trie, tokenizer, model, full_vocab):
    normal_batch = []
    adv_batch = []
    idxs = []
    for i in inp:
        if len(i.split(" ")) > 1:
            adv_batch.append(i)
            idxs.append(1)
        else:
            normal_batch.append(i)
            idxs.append(0)
    normal_res = predict_normal_batch(normal_batch, tokenizer, model)
    adv_res = predict_adv_batch(adv_batch, trie, tokenizer, model, full_vocab)
    res = []
    for i in idxs:
        if i == 0:
            res.append(normal_res.pop(0))
        else:
            res.append(adv_res.pop(0))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.ckpt)
    tokenizer = get_tokenizer(MODEL)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    # model.resize_token_embeddings(len(tokenizer))
    assert model.config.vocab_size == len(tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    full_vocab = list(range(len(tokenizer)))
    model.config.pad_token_id = tokenizer.eos_token_id
    logging.info(f"{tokenizer.truncation_side=}")

    train_dataset = CaptionDataset(tokenizer, f"{DIALOGCC_DATA_PATH}test.csv", test = False)
    trie = make_trie(train_dataset, tokenizer)
    dataset = CaptionDataset(tokenizer, f"{DIALOGCC_DATA_PATH}test.csv", test=True)


    if not os.path.exists(args.csv_path):
        os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
        pd.DataFrame(columns = ["idx", "pred"]).to_csv(args.csv_path, index=False)
    
    model.to("cuda")
    model.eval()
    for batch_idx in tqdm(range(0, len(dataset), BATCH_SIZE)):
        start = time.time()
        batch = [dataset[i] for i in range(batch_idx, min(batch_idx + BATCH_SIZE, len(dataset)))]
        print(f"time taken to load batch {batch_idx}: {time.time() - start}")
        # batch = dataset[batch_idx: batch_idx + BATCH_SIZE]
        start = time.time()
        completions = predict_batch(
            [b["full_text"] for b in batch], trie, tokenizer, model, full_vocab
        )
        print(f"time taken to predict batch {batch_idx}: {time.time() - start}")
        start = time.time()
        predictions_data = []
        for idx, line in enumerate(batch):
            predictions_data.append({
                "idx": line["idx"],
                "pred": completions[idx]
            })
        pd.DataFrame(predictions_data).to_csv(args.csv_path, mode='a', header=False, index=False)
        print(f"Time taken to write batch {batch_idx}: {time.time() - start}")
    # model.resize_token_embeddings(len(tokenizer))
    # model.load_state_dict(torch.load(CKPT))
