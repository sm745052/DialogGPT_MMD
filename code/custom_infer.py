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

MODEL = "microsoft/dialogpt-medium"
CKPT = "ckpt/dialogcc/checkpoint-1500"


CUSTOM_DIALOG: Dialog = Dialog(
    [
        Utterance(
            text="hey how are these ?",
            speaker="Anita",
            images=["im1"],
            captions=["image of a red dress"],
        ),
        Utterance(
            text="anita that is a really good dress",
            speaker="sam",
        ),
        Utterance(
            text="does the red color suit me ?",
            speaker="anita",
        ),
    ],
    idx = "custom"
)



if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(CKPT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    # model.resize_token_embeddings(len(tokenizer))
    dataset = CaptionDataset(tokenizer, [CUSTOM_DIALOG], test=True, custom = True)
    enc = dataset[24]
    print(tokenizer.decode(enc["input_ids"], skip_special_tokens = False))
    output = model.generate(
        input_ids = torch.tensor(enc["input_ids"]).unsqueeze(0),
        attention_mask = torch.tensor(enc["attention_mask"]).unsqueeze(0),
        max_length = 200,
        num_beams = 5,
        early_stopping = True,
    )
    print(tokenizer.decode(output[0], skip_special_tokens = False))

    # model.resize_token_embeddings(len(tokenizer))
    # model.load_state_dict(torch.load(CKPT))
