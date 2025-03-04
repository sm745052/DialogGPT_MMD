import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from chatas.code.utils.dataset import (
    DialogCCData,
    TEST_CONFIG,
    create_image_path_by_url,
    Dialog,
)
import torch
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, Any
from copy import deepcopy

DIALOGCC_DATA_PATH = "mnt/CHAT-AS-MULTIMODAL/data/DialogCC/"
IMAGES_DIR = "./mnt/tmp/"

IMAGE_TOKEN = "<|IMAGE|>"
MARKER_TOKEN = "<|MARK|>"

TOK_MAX_LENGTH = 256


def lower_dialog(dialog: Dialog) -> Dialog:
    for utt in dialog.utterances:
        utt.text = utt.text.lower()
        if utt.captions:
            utt.captions = [caption.lower() for caption in utt.captions]
        utt.speaker = utt.speaker.lower()
    return dialog


class CaptionDataset:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str = f"{DIALOGCC_DATA_PATH}train.csv",
        test: bool = False,
        custom: bool = False,
    ):
        self.tokenizer = tokenizer
        self.test = test
        self.custom = custom
        if self.test:
            if not self.custom:
                self.raw_data = DialogCCData(
                    path=data_path,
                    image_path_by_url=create_image_path_by_url(
                        f"{IMAGES_DIR}image_names", f"{IMAGES_DIR}images_n"
                    ),
                    **TEST_CONFIG,
                )
                # IDX = "te_bst:185__u11__s37"
                # self.raw_data = DialogCCData(
                #     path=([self.raw_data.dialog_suffix_by_id[IDX][0]], [self.raw_data.dialog_suffix_by_id[IDX][1]]),
                #     # to_split=True,
                # )
                # print("Test data loaded")
            else:
                self.raw_data = DialogCCData(
                    path=data_path,
                    to_split=True,
                )
        else:
            if not self.custom:
                self.raw_data = DialogCCData(
                    path=data_path,
                    image_path_by_url=create_image_path_by_url(
                        f"{IMAGES_DIR}image_names", f"{IMAGES_DIR}images_n"
                    ),
                    to_filter=True,
                    to_replace=True,
                    to_unroll=False,
                    min_images_per_dialog=1,
                    to_split=False,
                )
            else:
                self.raw_data = DialogCCData(
                    path=data_path,
                )
        self.raw_data = list(
            map(
                lambda x: (
                    lower_dialog(x[0]),
                    x[1].lower() if x[1] is not None else None,
                ),
                self.raw_data,
            )
        )

    def transform(self, conv: Dialog, suffix: str) -> Dict[str, Any]:
        utts = []
        for utt in conv.utterances[:-1]:
            captions = "".join(
                [
                    f"{IMAGE_TOKEN}{caption}{IMAGE_TOKEN}"
                    for caption in utt.captions
                ]
            ) if utt.captions else ""
            utts.append(f"{utt.speaker}: {captions}{utt.text}")
        captions = "".join(
            [
                f"{IMAGE_TOKEN}{caption}{IMAGE_TOKEN}"
                for caption in conv.utterances[-1].captions
            ]
        ) if conv.utterances[-1].captions else ""
        if self.test:
            utts.append(f"{conv.utterances[-1].speaker}: {captions}{conv.utterances[-1].text}")
            text = self.tokenizer.eos_token.join(utts)
            full_text = text
            if not text.endswith(self.tokenizer.eos_token) and not text.endswith(IMAGE_TOKEN) and text[-1] != " ":
                last_incomplete_word = text.split(" ")[-1].split(IMAGE_TOKEN)[-1].split(self.tokenizer.eos_token)[-1]
                # print(f"{last_incomplete_word=}")
                text = text[:-len(last_incomplete_word)].rstrip()
                # print(f"{text=}")
                # print(f"{suffix=}")
                # suffix = f"{last_incomplete_word}{suffix}"
                # print(f"{suffix=}")

            enc = self.tokenizer(text, truncation=True, max_length=TOK_MAX_LENGTH)
            # print(f"{suffix=}")
            # print(
            #     "hello",
            #     {
            #         "input_ids": enc["input_ids"],
            #         "attention_mask": enc["attention_mask"],
            #         "suffix": suffix,
            #     },
            # )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "suffix": suffix,
                "prefix": text,
                "full_text": full_text,
                "idx": conv.idx,
            }
        else:
            utts.append(
                f"{conv.utterances[-1].speaker}: {captions}{MARKER_TOKEN}{conv.utterances[-1].text}{suffix}{self.tokenizer.eos_token}"
            )
            text = self.tokenizer.eos_token.join(utts)
            enc = self.tokenizer(text, truncation=True, max_length=TOK_MAX_LENGTH+1)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            if self.tokenizer.convert_tokens_to_ids(MARKER_TOKEN) in input_ids:
                marker_pos = input_ids.index(self.tokenizer.convert_tokens_to_ids(MARKER_TOKEN))
            else:
                marker_pos = 0
            input_ids = input_ids[:marker_pos] + input_ids[marker_pos+1:]
            attention_mask = attention_mask[:marker_pos] + attention_mask[marker_pos+1:]
            labels = deepcopy(input_ids)
            labels[:marker_pos] = [-100] * marker_pos
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    def __getitem__(self, idx) -> Dict[str, Any]:
        if self.test:
            # print(f"{self.raw_data[idx]=}")
            return self.transform(*self.raw_data[idx])
        rng = np.random.RandomState(seed=42)
        conv, _ = self.raw_data[idx]
        unrolled = conv.unroll()
        unrolled_filter_min_one_image = list(
            filter(
                lambda x: sum([len(u.images) for u in x.utterances]) >= 1,
                unrolled,
            )
        )
        if len(unrolled_filter_min_one_image) == 0:
            print(
                f"This shouldnot happen, couldnot find a unroll of data that has at least one image \n DialogId: {conv.idx}"
            )
            return self.__getitem__(rng.randint(len(self)))
        splits = conv.create_splits()
        random_split = splits[rng.randint(len(splits))]
        return self.transform(*random_split)

    def __len__(self):
        return len(self.raw_data)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/dialogpt-small", truncation_side = "left")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [IMAGE_TOKEN, MARKER_TOKEN]}
    )
    tokenizer.pad_token = tokenizer.eos_token
    dataset = CaptionDataset(tokenizer, test=True)
    # print(dataset[14])
    ele = dataset[14]
    print(tokenizer.decode(ele["input_ids"]))
    print(ele["suffix"])
