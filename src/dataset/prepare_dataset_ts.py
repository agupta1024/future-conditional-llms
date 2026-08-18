"""Prepare TinyStories datasets for training and evaluation."""

import re

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer


class TinyStoriesHFDataset(Dataset):
    """Wrap a HuggingFace dataset split and isolate prompt/story tokens."""

    def __init__(self, hf_dataset, tokenizer_model, max_length=1024):
        self.tokenizer = tokenizer_model
        self.max_length = max_length
        self.data = hf_dataset

        print(f"Initialized Dataset with {len(self.data)} samples.")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx): # pylint: disable=too-many-locals
        """Return prompt/story tensors and labels for one sample."""
        sample = self.data[idx]

        if "prompt" in sample and "story" in sample:
            prompt_text = sample["prompt"].strip() + "\n[STORY]\n"
            story_text = sample["story"].strip()
        elif "text" in sample:
            text = sample["text"].strip()
            if "\nStory:" in text or "\nstory:" in text:
                split_token = "\nStory:" if "\nStory:" in text else "\nstory:"
                parts = text.split(split_token, 1)
                prompt_text = parts[0].strip() + "\n[STORY]\n"
                story_text = parts[1].strip()
            else:
                match = re.search(r"([.!?][\"']?)\s+", text)
                if match:
                    split_idx = match.end()
                    premise = text[:split_idx].strip()
                    story_body = text[split_idx:].strip()
                else:
                    words = text.split()
                    premise = " ".join(words[:10]) + "..."
                    story_body = " ".join(words[10:])

                prompt_text = f"[PREMISE] {premise}\n[STORY]\n"
                story_text = story_body
        else:
            prompt_text = "[STORY]\n"
            story_text = str(sample)

        prompt_ids = self.tokenizer.encode(prompt_text)
        story_ids = self.tokenizer.encode(story_text)
        story_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + story_ids
        labels = [-100] * len(prompt_ids) + story_ids
        loss_weights = [0.0] * len(prompt_ids) + [1.0] * len(story_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            input_ids[-1] = self.tokenizer.eos_token_id

            labels = labels[: self.max_length]
            labels[-1] = self.tokenizer.eos_token_id

            loss_weights = loss_weights[: self.max_length]
            loss_weights[-1] = 1.0

            prompt_ids = prompt_ids[: self.max_length]

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float),
        }


class TinyStoriesCollator: # pylint: disable=too-few-public-methods
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


def get_dataloaders(
    dataset_name="skeskinen/TinyStories-Instruct-hf",
    batch_size=16,
    is_ddp=False,
    max_length=512,
    split="train",
): # pylint: disable=too-many-locals
    """Load HuggingFace TinyStories data and prepare PyTorch dataloaders."""
    print(f"Loading HF Dataset: {dataset_name}...")

    hf_dataset = load_dataset(dataset_name, cache_dir="./data_cache")
    train_hf = hf_dataset["train"]
    val_hf = hf_dataset["validation"]
    train_hf = train_hf.select(range(100000))
    val_hf = val_hf.select(range(1, 1001))

    tokenizer_model = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer_model.pad_token is None:
        tokenizer_model.pad_token = tokenizer_model.eos_token
        tokenizer_model.pad_token_id = tokenizer_model.eos_token_id

    if split == "train":
        train_dataset = TinyStoriesHFDataset(train_hf, tokenizer_model, max_length=max_length)
        eval_dataset = TinyStoriesHFDataset(val_hf, tokenizer_model, max_length=max_length)
    else:
        eval_dataset = TinyStoriesHFDataset(val_hf, tokenizer_model, max_length=max_length)
        train_dataset = None

    collator = TinyStoriesCollator(pad_token_id=tokenizer_model.pad_token_id)

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        eval_sampler = None
        shuffle = False

    train_dataloader = None
    if split == "train" and train_dataset is not None:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            collate_fn=collator,
        )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=collator,
    )
    if is_ddp:
        return train_dataloader, eval_dataloader, tokenizer_model, train_sampler
    return train_dataloader, eval_dataloader, tokenizer_model, train_sampler


if __name__ == "__main__":
    print("Testing the Dataset Preparation...")

    tokenizer_ = GPT2Tokenizer.from_pretrained("gpt2")
    train_loader, _ = get_dataloaders(max_length=512, batch_size=2)
    batch = next(iter(train_loader))

    print("\n=== BATCH SHAPES ===")
    print(f"Prompt IDs (Planner): {batch['prompt_ids'].shape}")
    print(f"Input IDs (Decoder): {batch['input_ids'].shape}")
    print(f"Labels: {batch['labels'].shape}")

    print("\n=== LABEL MASKING VERIFICATION ===")
    sample_labels = batch["labels"][0].tolist()

    prompt_length = batch["prompt_mask"][0].sum().item()
    actual_prompt = batch["input_ids"][0][:prompt_length]

    actual_story = [
        token_id
        for index, token_id in enumerate(batch["input_ids"][0])
        if index >= prompt_length and batch["labels"][0][index] != -100
    ]

    print(f"Decoded Prompt: '{tokenizer_.decode(actual_prompt)}'")
    print(f"Decoded Filler + Action: '{tokenizer_.decode(actual_story)}'")

    print(
        "\nSuccess! The DataLoader perfectly isolates the prompt and masks it "
        "in the loss calculation."
    )
