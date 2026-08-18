"""Train GPT-2 style models on the TinyStories, treasure hunt, and Blocksworld tasks."""

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_dataset_th import get_dataloaders as get_th_dataloaders
from ..dataset.prepare_dataset_ts import get_dataloaders as get_ts_dataloaders

def train(dataset_name: str): # pylint: disable=too-many-locals,too-many-statements
    """Train a GPT-2 style model on the specified dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Base Training on: {device}")
    num_epochs = 30
    learning_rate = 1e-4

    base_working_model = 'gpt2_1024-l' # for bw, gpt2_512-l is used for treasure hunt and tinystories
    ds_name = dataset_name
    stage = "base"
    model_map = {
        "gpt2_512-l": [512],
        "gpt2_1024-l": [1024],
    }

    dataset_config = get_dataset_config(name=ds_name)
    train_path = dataset_config.get("train_path", "")
    print(f"Loading {ds_name} dataset from {train_path}...")
    validation_path = dataset_config.get("val_path", "")
    batch_size = dataset_config.get("batch_size", 16)
    batch_size = min(batch_size, 16)

    print("Preparing Datasets...")
    if "blocksworld" in ds_name:
        dataloader_cls = get_bw_dataloaders
    elif ds_name == "treasure_hunt":
        dataloader_cls = get_th_dataloaders
    else:
        dataloader_cls = get_ts_dataloaders

    if ds_name != "tinystories":
        train_loader, eval_loader, _, _ = dataloader_cls(
            train_path=train_path,
            eval_path=validation_path,
            batch_size=batch_size,
            tokenizer_path=dataset_config.get("tokenizer_path", ""),
            is_ddp=False
        )
    else:
        train_loader, eval_loader, _, _ = dataloader_cls(
            dataset_name="skeskinen/TinyStories-Instruct-hf",
            batch_size=batch_size,
            is_ddp=False,
            max_length=512,
        )

    _, model = get_model_and_tokenizer(
        working_model=base_working_model,
        max_seq_length=model_map[base_working_model][0],
        load_scratch=True,
        dataset_name=ds_name,
        load_stage=stage,
        custom_ar=True,
        vocab_size=dataset_config.get("vocab_size", 50257),
        tokenizer_path=dataset_config.get("tokenizer_path", "")
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    outdir = f"./{ds_name}_{stage}_model_{base_working_model}"
    os.makedirs(outdir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for _, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["loss_weights"].to(device)

            if input_ids.size(1) > model.config.n_positions:
                input_ids = input_ids[:, :model.config.n_positions]
                labels = labels[:, :model.config.n_positions]
                weights = weights[:, :model.config.n_positions]
            input_ids = torch.clamp(input_ids, min=0, max=model.config.vocab_size - 1)

            logits = model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            weighted_loss = loss * shift_weights.view(-1)

            active_tokens = (shift_weights.view(-1) > 0).sum()
            loss = weighted_loss.sum() / active_tokens
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.3f}",
            })
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(outdir, f"checkpoint_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        model.eval()
        total_eval_loss = 0

        print("Running Evaluation...")
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                weights = batch["loss_weights"].to(device)

                if input_ids.size(1) > model.config.n_positions:
                    input_ids = input_ids[:, :model.config.n_positions]
                    labels = labels[:, :model.config.n_positions]
                    weights = weights[:, :model.config.n_positions]
                input_ids = torch.clamp(input_ids, min=0, max=model.config.vocab_size - 1)
                logits = model(input_ids)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_weights = weights[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, model.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                weighted_loss = loss * shift_weights.view(-1)
                active_tokens = (shift_weights.view(-1) > 0).sum()
                loss = weighted_loss.sum() / active_tokens

                total_eval_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_eval_loss = total_eval_loss / len(eval_loader)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Loss:  {avg_eval_loss:.4f}")

    print(f"\nSaving From-Scratch Baseline to {outdir}...")
    torch.save(model.state_dict(), f"{outdir}/base_model.pt")
    print("Phase 1: Baseline Training Complete!")

if __name__ == "__main__":
    train(dataset_name="blocksworld_sub")
