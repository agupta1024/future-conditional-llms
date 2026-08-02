"""Train GPT-2 style models on the TinyStories, treasure hunt, and Blocksworld tasks."""

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import wandb

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_dataset_th import get_dataloaders as get_th_dataloaders
from ..dataset.prepare_dataset_ts import get_dataloaders as get_ts_dataloaders
from .utils import init_wandb

def train(dataset_name: str): # pylint: disable=too-many-locals,too-many-statements
    """Train a GPT-2 style model on the specified dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Base Training on: {device}")
    num_epochs = 30
    learning_rate = 1e-4

    base_working_model = 'gpt2_512_512'
    ds_name = dataset_name
    stage = "base"
    model_map = {
        "gpt2_hf": [2048, 768],
        "gpt2_hf_1024": [1024, 768],
        "gpt2_512_1024": [1024, 512],
        "gpt2_512_512": [512, 512],
        "hf_gpt2_512_512": [512, 512],
    }
    _, model = get_model_and_tokenizer(
        working_model=base_working_model,
        hidden_dim=model_map[base_working_model][1],
        max_seq_length=model_map[base_working_model][0],
        load_scratch=True,
        dataset_name=ds_name,
        load_stage=stage,
        custom_ar=True,
    )
    model.to(device)

    dataset_config = get_dataset_config(name=ds_name)
    train_path = dataset_config.get("train_path", "")
    print(f"Loading {ds_name} dataset from {train_path}...")
    validation_path = dataset_config.get("val_path", "")
    batch_size = dataset_config.get("batch_size", 16)

    # 3. Load Datasets
    print("Preparing Datasets...")
    if ds_name == "blocksworld":
        train_dataloader, eval_dataloader = get_bw_dataloaders(
            train_path=train_path,
            eval_path=validation_path,
            batch_size=batch_size,
        )
    elif ds_name == "treasure_hunt":
        train_dataloader, eval_dataloader = get_th_dataloaders(
            train_path=train_path,
            eval_path=validation_path,
            batch_size=batch_size,
        )
    else:
        train_dataloader, eval_dataloader = get_ts_dataloaders(
            dataset_name="skeskinen/TinyStories-Instruct-hf",
            batch_size=batch_size,
            is_ddp=False,
            max_length=512,
        )

    wandb_run_name = f"{stage}-{ds_name}-{base_working_model}"
    init_wandb(
        model_name=base_working_model,
        stage=stage,
        ds_name=ds_name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        ds_size=len(train_dataloader.dataset),
        run_name=wandb_run_name,
    )
    # 4. Optimizer and Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    outdir = f"./{ds_name}_{stage}_model_{base_working_model}"
    os.makedirs(outdir, exist_ok=True)
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        print(f"\n=== Epoch {epoch + 1} / {num_epochs} ===")
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["loss_weights"].to(device)
            if input_ids.size(1) > model.config.n_positions:
                input_ids = input_ids[:, -model.config.n_positions:]
                labels = labels[:, -model.config.n_positions:]
                weights = weights[:, -model.config.n_positions:]
            input_ids = torch.clamp(input_ids, min=0, max=model.config.vocab_size - 1)
            logits = model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            raw_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            weighted_loss = raw_loss * shift_weights.view(-1)

            active_tokens = (shift_weights.view(-1) > 0).sum()
            loss = weighted_loss.sum() / active_tokens

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1
            total_train_loss += loss.item()
            if step % 50 == 0:
                print(f"Step {step}/{len(train_dataloader)} | Loss: {loss.item():.4f}\
                    | LR: {scheduler.get_last_lr()[0]:.2e}")
            wandb.log({
                "train/loss": loss.item(),
                "epoch": epoch + 1
            }, step=global_step)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.3f}",
            })

        print(f"Epoch {epoch+1}/{num_epochs} complete. Saving checkpoint...")
        checkpoint_path = f"{outdir}/checkpoint-{global_step}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}\n")


        model.eval()
        total_eval_loss = 0

        print("Running Evaluation...")
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                weights = batch["loss_weights"].to(device)
                if input_ids.size(1) > model.config.n_positions:
                    input_ids = input_ids[:, -model.config.n_positions:]
                    labels = labels[:, -model.config.n_positions:]
                    weights = weights[:, -model.config.n_positions:]
                input_ids = torch.clamp(input_ids, min=0, max=model.config.vocab_size - 1)

                logits = model(input_ids)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_weights = weights[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, model.config.vocab_size),
                    shift_labels.view(-1)
                )

                total_eval_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        wandb.log({
            "train/avg_loss": avg_train_loss,
            "eval/avg_loss": avg_eval_loss,
            "epoch": epoch + 1,
        }, step=global_step)
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Loss:  {avg_eval_loss:.4f}")

    save_path = f"{outdir}/base_model.pt"
    print(f"\nSaving Domain Adapted model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Phase 1: Domain Adaptation Complete!")

if __name__ == "__main__":
    train(dataset_name="tinystories")  # treasure_hunt, blocksworld
