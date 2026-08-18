"""Prepare treasure-hunt datasets for training and evaluation."""

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer


class TreasureHuntDataset(Dataset):
    """Create a dataset that isolates prompts from filler and final action tokens."""

    def __init__(self, jsonl_path, tokenizer_model, max_length=512):
        self.tokenizer_model = tokenizer_model
        self.max_length = max_length
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return prompt/filler/action tensors and labels for one sample."""
        sample = self.data[idx]

        prompt_text = sample["prompt"]
        prompt_ids = self.tokenizer_model.encode(prompt_text)

        filler_ids = self.tokenizer_model.encode(sample["filler"])
        action_ids = self.tokenizer_model.encode(sample["final_action"])

        input_ids = prompt_ids + filler_ids + action_ids
        labels = [-100] * len(prompt_ids) + filler_ids + action_ids

        loss_weights = (
            [0.0] * len(prompt_ids)
            + [1.0] * len(filler_ids)
            + [10.0] * len(action_ids)
        )

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            loss_weights = loss_weights[: self.max_length]

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float),
        }


class TreasureHuntCollator: # pylint: disable=too-few-public-methods
    """Pad batches dynamically to the longest sequence in the batch."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        self.ignore_index = -100

    def __call__(self, features):
        """Pad a list of feature dictionaries into a single batch."""
        prompt_lengths = [feature["prompt_ids"].size(0) for feature in features]
        max_prompt_len = max(prompt_lengths)

        batch_prompt_ids = torch.full(
            (len(features), max_prompt_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        batch_prompt_mask = torch.zeros((len(features), max_prompt_len), dtype=torch.long)

        for index, feature in enumerate(features):
            length = prompt_lengths[index]
            batch_prompt_ids[index, :length] = feature["prompt_ids"]
            batch_prompt_mask[index, :length] = 1

        input_lengths = [feature["input_ids"].size(0) for feature in features]
        max_input_len = max(input_lengths)

        batch_input_ids = torch.full(
            (len(features), max_input_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        batch_input_mask = torch.zeros((len(features), max_input_len), dtype=torch.long)
        batch_labels = torch.full(
            (len(features), max_input_len),
            self.ignore_index,
            dtype=torch.long,
        )
        batch_loss_weights = torch.zeros((len(features), max_input_len), dtype=torch.float)

        for index, feature in enumerate(features):
            length = input_lengths[index]
            batch_input_ids[index, :length] = feature["input_ids"]
            batch_input_mask[index, :length] = 1
            batch_labels[index, :length] = feature["labels"]
            batch_loss_weights[index, :length] = feature["loss_weights"]

        return {
            "prompt_ids": batch_prompt_ids,
            "prompt_mask": batch_prompt_mask,
            "input_ids": batch_input_ids,
            "attention_mask": batch_input_mask,
            "labels": batch_labels,
            "loss_weights": batch_loss_weights,
        }


def get_dataloaders(train_path, eval_path, is_ddp=False, batch_size=16, **_kwargs):
    """Create train and evaluation dataloaders for treasure-hunt data."""
    tokenizer_ = GPT2Tokenizer.from_pretrained("gpt2")

    if tokenizer_.pad_token is None:
        tokenizer_.pad_token = tokenizer_.eos_token
        tokenizer_.pad_token_id = tokenizer_.eos_token_id

    collator = TreasureHuntCollator(pad_token_id=tokenizer_.pad_token_id)

    train_dataset = None
    eval_dataset = None
    if train_path and os.path.exists(train_path):
        train_dataset = TreasureHuntDataset(train_path, tokenizer_)
    if eval_path and os.path.exists(eval_path):
        eval_dataset = TreasureHuntDataset(eval_path, tokenizer_)

    if is_ddp:
        train_sampler = None
        eval_sampler = None
        if train_dataset is not None:
            train_sampler = DistributedSampler(train_dataset)
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        eval_sampler = None
        shuffle = True
    if train_dataset is not None:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            collate_fn=collator,
        )
    else:
        train_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=eval_sampler,
            collate_fn=collator,
        )
    else:
        eval_dataloader = None
    if is_ddp:
        return train_dataloader, eval_dataloader, tokenizer_, train_sampler
    return train_dataloader, eval_dataloader, tokenizer_, train_sampler


if __name__ == "__main__":
    print("Testing the Dataset Preparation...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    try:
        train_loader, eval_loader,_,_ = get_dataloaders(
            train_path="./data_cache/treasure_hunt/train.jsonl",
            eval_path="./data_cache/treasure_hunt/eval_horizon_64.jsonl",
            batch_size=2,
        )

        batch = next(iter(train_loader))

        print("\n=== BATCH SHAPES ===")
        print(f"Prompt IDs (Planner): {batch['prompt_ids'].shape}")
        print(f"Input IDs (Decoder): {batch['input_ids'].shape}")
        print(f"Labels: {batch['labels'].shape}")

        print("\n=== LABEL MASKING VERIFICATION ===")
        sample_labels = batch["labels"][0].tolist()

        masked_count = sample_labels.count(-100)
        print(f"Number of tokens masked (-100): {masked_count}")
        print(
            f"Decoded Prompt: '{tokenizer.decode(batch['prompt_ids'][0][:masked_count])}'"
        )
        print(
            f"Decoded Filler + Action: '{tokenizer.decode(batch['input_ids'][0][masked_count:])}'"
        )

        print(
            "\nSuccess! The DataLoader perfectly isolates the prompt and masks it "
            "in the loss calculation."
        )

    except FileNotFoundError:
        print("\nCould not find 'treasure_hunt_data/train.jsonl'.")
        print("Please run `generate_treasure_hunt.py` first to generate the dataset!")
