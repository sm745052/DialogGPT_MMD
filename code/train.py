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

MODEL = "microsoft/dialogpt-medium"
# CKPT = MODEL

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
OUTPUT_FOLDER = "mnt/sm745052/tmp-ckpts/dialogpt_dcc"

WANDB_FOLDER = "mnt/sm745052/tmp-logs/"


wandb.init(
    dir=WANDB_FOLDER,
)


@dataclass
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        print(
            f"Will use {self.tokenizer.pad_token} as padding token and padding side is {self.tokenizer.padding_side}"
        )

    def __call__(self, examples: List[Dict[str, torch.Tensor]]):
        batch = {}
        batch["input_ids"] = []
        batch["attention_mask"] = []
        batch["labels"] = []
        for example in examples:
            batch["input_ids"].append(example["input_ids"])
            batch["attention_mask"].append(example["attention_mask"])
            batch["labels"].append(example["labels"])
            # print(f"{example=}")
            input_ids_text = self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)
            # labels_text = self.tokenizer.decode(example["labels"], skip_special_tokens=False)
            # print(f"{input_ids_text=}")
            # print(f"{labels_text=}")
        # print(f"before padding {batch=}")
        batch = self._pad_batch(batch)
        # print(f"after padding {batch=}")
        batch = self._to_tensors(batch)
        return batch

    def _to_tensors(
        self, batch: Dict[str, List[Union[List[int], torch.Tensor]]]
    ):
        return {key: torch.tensor(value) for key, value in batch.items()}

    def _pad_batch(self, batch: Dict[str, torch.Tensor]):
        max_length = max(len(x) for x in batch["input_ids"])
        # max_length = TOKENIZER_MAX_LENGTH
        # print(f"{max_length=}")
        padded_batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i]
            attention_mask = batch["attention_mask"][i]
            labels = batch["labels"][i]
            padding_length = max_length - len(input_ids)
            if self.tokenizer.padding_side == "right":
                padded_input_ids = (
                    input_ids + [self.tokenizer.pad_token_id] * padding_length
                )
                padded_attention_mask = attention_mask + [0] * padding_length
                padded_labels = labels + [-100] * padding_length
            else:
                padded_input_ids = [
                    self.tokenizer.pad_token_id
                ] * padding_length + input_ids
                padded_attention_mask = [0] * padding_length + attention_mask
                padded_labels = [-100] * padding_length + labels
            padded_batch["input_ids"].append(padded_input_ids)
            padded_batch["attention_mask"].append(padded_attention_mask)
            padded_batch["labels"].append(padded_labels)
        # print(f"{padded_batch=}")
        return padded_batch


if __name__ == "__main__":
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=False,
        truncation_side="left",
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.add_special_tokens({"additional_special_tokens": [MARKER_TOKEN]})
    train_dataset = CaptionDataset(
        tokenizer, data_path=f"{DIALOGCC_DATA_PATH}train.csv", test=False
    )
    # print(len(train_dataset))
    val_dataset = CaptionDataset(
        tokenizer, data_path=f"{DIALOGCC_DATA_PATH}validation.csv", test=False
    )
    data_collator = CustomDataCollator(tokenizer)
    training_args = TrainingArguments(
        output_dir=OUTPUT_FOLDER,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        logging_dir="logs",
        logging_first_step=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(OUTPUT_FOLDER) 
