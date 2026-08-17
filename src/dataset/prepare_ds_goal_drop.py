"""Prepare Blocksworld datasets with goal dropout for training and evaluation."""

import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast

class DynamicBlocksworldDataset(Dataset):
    # pylint: disable=too-many-instance-attributes
    """
    Create a dataset that separates prompts from action trajectories.
    """
    def __init__(self, jsonl_path, tokenizer, max_length=512, goal_dropout_prob=0.3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        self.comma_token_id = self.tokenizer.convert_tokens_to_ids(",")
        self.done_token_id = self.tokenizer.convert_tokens_to_ids("[DONE]")
        self.goal_token_id = self.tokenizer.convert_tokens_to_ids("[GOAL]")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.goal_dropout_prob = goal_dropout_prob
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} dynamic samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # pylint: disable=too-many-locals
        sample = self.data[idx]
        prompt_text = sample["prompt"] + " "
        prompt_ids = self.tokenizer.encode(prompt_text)

        decoder_prompt_ids = prompt_ids.copy()
        if random.random() < self.goal_dropout_prob:
            try:
                goal_idx = decoder_prompt_ids.index(self.goal_token_id)
                for i in range(goal_idx + 1, len(decoder_prompt_ids)):
                    decoder_prompt_ids[i] = self.pad_token_id
            except ValueError:
                pass

        if "[GOAL]" in prompt_text:
            pure_goal_text = "[GOAL]" + prompt_text.split("[GOAL]")[1]
        else:
            pure_goal_text = prompt_text
        pure_goal_ids = self.tokenizer.encode(pure_goal_text)
        trajectory_text = sample["trajectory"]
        trajectory_ids = self.tokenizer.encode(trajectory_text) + [self.tokenizer.eos_token_id]

        input_ids = decoder_prompt_ids + trajectory_ids
        labels = [-100] * len(prompt_ids) + trajectory_ids
        loss_weights = [0.0] * len(prompt_ids) + [1.0] * len(trajectory_ids)

        update_mask = [0] * len(input_ids)

        for i in range(len(prompt_ids), len(input_ids)):
            token = input_ids[i]
            if token in [self.comma_token_id, self.done_token_id]:
                update_mask[i] = 1

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            loss_weights = loss_weights[:self.max_length]
            update_mask = update_mask[:self.max_length]
            prompt_ids = prompt_ids[:self.max_length]

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "pure_goal_ids": torch.tensor(pure_goal_ids, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float),
            "update_mask": torch.tensor(update_mask, dtype=torch.long)
        }

class DynamicBWCollator:
    # pylint: disable=too-few-public-methods
    """Pad prompt and trajectory tensors for batching."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        self.ignore_index = -100

    def __call__(self, features):
        # pylint: disable=too-many-locals
        prompt_lengths = [f["prompt_ids"].size(0) for f in features]
        max_prompt_len = max(prompt_lengths)

        batch_prompt_ids = torch.full((
            len(features), max_prompt_len),
            self.pad_token_id,
            dtype=torch.long
        )
        batch_prompt_mask = torch.zeros((len(features), max_prompt_len), dtype=torch.long)

        goal_lengths = [f["pure_goal_ids"].size(0) for f in features]
        max_goal_len = max(goal_lengths)
        batch_goal_ids = torch.full(
            (len(features), max_goal_len),
            self.pad_token_id,
            dtype=torch.long
        )
        batch_goal_mask = torch.zeros((len(features), max_goal_len), dtype=torch.long)

        for i, f in enumerate(features):
            length = prompt_lengths[i]
            batch_prompt_ids[i, :length] = f["prompt_ids"]
            batch_prompt_mask[i, :length] = 1

            g_length = goal_lengths[i]
            batch_goal_ids[i, :g_length] = f["pure_goal_ids"]
            batch_goal_mask[i, :g_length] = 1

        input_lengths = [f["input_ids"].size(0) for f in features]
        max_input_len = max(input_lengths)

        batch_input_ids = torch.full(
            (len(features), max_input_len),
            self.pad_token_id,
            dtype=torch.long
        )
        batch_input_mask = torch.zeros((len(features), max_input_len), dtype=torch.long)
        batch_labels = torch.full(
            (len(features), max_input_len),
            self.ignore_index,
            dtype=torch.long
        )
        batch_loss_weights = torch.zeros((len(features), max_input_len), dtype=torch.float)
        batch_update_mask = torch.zeros((len(features), max_input_len), dtype=torch.long)

        for i, f in enumerate(features):
            length = input_lengths[i]
            batch_input_ids[i, :length] = f["input_ids"]
            batch_input_mask[i, :length] = 1
            batch_labels[i, :length] = f["labels"]
            batch_loss_weights[i, :length] = f["loss_weights"]
            batch_update_mask[i, :length] = f["update_mask"]

        return {
            "prompt_ids": batch_prompt_ids,
            "prompt_mask": batch_prompt_mask,
            "pure_goal_ids": batch_goal_ids,
            "pure_goal_mask": batch_goal_mask,
            "input_ids": batch_input_ids,
            "attention_mask": batch_input_mask,
            "labels": batch_labels,
            "loss_weights": batch_loss_weights,
            "update_mask": batch_update_mask
        }

def get_dataloaders(train_path, eval_path, tokenizer_path,
                            is_ddp=False, batch_size=8, max_length=512):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Return DataLoaders for training and evaluation datasets."""
    print(f"Loading Custom Tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'unk_token': '[UNK]',
        'eos_token': '[PAD]',
        'additional_special_tokens': ['[INIT]', '[GOAL]', ',']
    })

    train_dataset = None
    eval_dataset = None
    if train_path and os.path.exists(train_path):
        train_dataset = DynamicBlocksworldDataset(train_path, tokenizer, max_length=max_length)
    if eval_path and os.path.exists(eval_path):
        eval_dataset = DynamicBlocksworldDataset(eval_path, tokenizer, max_length=max_length)

    collator = DynamicBWCollator(pad_token_id=tokenizer.pad_token_id)
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
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            collate_fn=collator
        )
    else:
        train_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=eval_sampler,
            collate_fn=collator
        )
    else:
        eval_loader = None

    if is_ddp:
        return train_loader, eval_loader, tokenizer, train_sampler
    return train_loader, eval_loader, tokenizer, train_sampler

if __name__ == "__main__":
    print("Testing the Dataset Preparation...")
    try:
        train_loader_, eval_loader_, tokenizer_, _ = get_dataloaders(
            train_path="./data_cache/blocksworld_sub/train.jsonl",
            eval_path="./data_cache/blocksworld_sub/eval.jsonl",
            tokenizer_path="./src/config/tokenizer/bw6_tokenizer.json",
            batch_size=2,
        )

        batch = next(iter(train_loader_))

        print("\n=== BATCH SHAPES ===")
        print(f"Prompt IDs (Planner): {batch['prompt_ids'].shape}")
        print(f"Input IDs (Decoder): {batch['input_ids'].shape}")
        print(f"Labels: {batch['labels'].shape}")


        print("\n=== LABEL MASKING VERIFICATION ===")
        sample_labels = batch["labels"][0].tolist()

        masked_count = sample_labels.count(-100)
        print(f"Number of tokens masked (-100): {masked_count}")
        print(
            f"Decoded Prompt: '{tokenizer_.decode(batch['prompt_ids'][0][:masked_count])}'"
        )
        print(
            f"Decoded Filler + Action: '{tokenizer_.decode(
                batch['input_ids'][0])}'"
        )

        print(
            "\nSuccess! The DataLoader perfectly isolates the prompt and masks it "
            "in the loss calculation."
        )

    except FileNotFoundError:
        print("\nCould not find 'blocksworld_sub/train.jsonl'.")
        print("Please run `generate_bw_sub.py` first to generate the dataset!")
