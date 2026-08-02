"""Writer Model Training Script for TinyStories, Treasure hunt and Blocksworld Datasets.
DDP enabled."""

import gc
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import set_seed
import wandb

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_dataset_th import get_dataloaders as get_th_dataloaders
from ..dataset.prepare_dataset_ts import get_dataloaders as get_ts_dataloaders
from .utils import init_wandb

# Set environment variables right after imports (or in your main function)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Clean up the process group at the end."""
    dist.destroy_process_group()

# --- Validation Loop Function ---
def run_validation(model, val_dataloader, device):
    # pylint: disable=too-many-locals
    """Run validation on the model using the provided dataloader."""
    model.eval()
    val_loss = 0.0
    print("Running Evaluation...")
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch["labels"].to(device)

            prompt_ids = batch.get('prompt_ids', input_ids).to(device)
            prompt_mask = batch.get('prompt_mask', attention_mask).to(device)
            if input_ids.size(1) > model.planner.encoder.config.n_positions:
                input_ids = input_ids[:, -model.planner.encoder.config.n_positions:]
                attention_mask = attention_mask[:, -model.planner.encoder.config.n_positions:]
                labels = labels[:, -model.planner.encoder.config.n_positions:]
            input_ids = torch.clamp(input_ids, min=0,
                                    max=model.planner.encoder.config.vocab_size - 1)

            with torch.no_grad():
                pred_future, _, _ = model.planner(prompt_ids, prompt_mask)
            logits = model(input_ids, pred_future)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, model.planner.encoder.config.vocab_size),
                shift_labels.view(-1)
            )

            val_loss += loss.item()

    avg_loss = val_loss / len(val_dataloader.dataset)
    return avg_loss


# --- Main Training Script ---
def train(dataset_name="tinystories"):
    # pylint: disable=too-many-locals, too-many-statements
    """Main training loop for the writer model."""
    torch.cuda.empty_cache()
    gc.collect()
    local_rank = setup_ddp()
    is_main_process = local_rank == 0
    device = torch.device(f"cuda:{local_rank}")
    print(f"Training on {device}")
    set_seed(42)

    base_working_model = 'gpt2_512_512'
    ds_name = dataset_name
    stage = "writer"
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
        load_scratch=False,
        dataset_name=ds_name,
        load_stage=stage,
        custom_ar=True,
        film=True,
    )
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dataset_config = get_dataset_config(name=ds_name, stage=stage)
    train_path = dataset_config.get("train_path", "")
    print(f"Loading {ds_name} dataset from {train_path}...")
    validation_path = dataset_config.get("val_128_path", "")
    batch_size = dataset_config.get("batch_size", 32)

    print("Preparing Datasets...")
    if ds_name == "blocksworld":
        train_dataloader, val_dataloader, train_sampler = get_bw_dataloaders(
            train_path=train_path,
            eval_path=validation_path,
            batch_size=batch_size,
            is_ddp=True,
        )
    elif ds_name == "treasure_hunt":
        train_dataloader, val_dataloader, train_sampler = get_th_dataloaders(
            train_path=train_path,
            eval_path=validation_path,
            batch_size=batch_size,
            is_ddp=True,
        )
    else:
        train_dataloader, val_dataloader, train_sampler = get_ts_dataloaders(
            dataset_name="skeskinen/TinyStories-Instruct-hf",
            batch_size=batch_size,
            max_length=512,
            is_ddp=True
        )
    learning_rate = 1e-4
    num_epochs = 30

    if is_main_process:
        wandb_run_name= f"{stage}-{ds_name}-{base_working_model}"
        init_wandb(
            model_name=base_working_model,
            stage=stage,
            ds_name=ds_name,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            ds_size=len(train_dataloader.dataset),
            run_name=wandb_run_name
        )

    outdir = f'./{ds_name}_{stage}_model_{base_working_model}'
    os.makedirs(outdir, exist_ok=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=learning_rate, weight_decay=1e-4)

    global_step = 0
    print("Beginning Training Loop...")
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.module.train()
        model.module.planner.eval()
        epoch_train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for _, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch["labels"].to(device)
            weights = batch["loss_weights"].to(device)

            prompt_ids = batch.get('prompt_ids', input_ids).to(device)
            prompt_mask = batch.get('prompt_mask', attention_mask).to(device)
            if input_ids.size(1) > model.module.planner.encoder.config.n_positions:
                input_ids = input_ids[:, -model.module.planner.encoder.config.n_positions:]
                attention_mask = attention_mask[:,
                                    -model.module.planner.encoder.config.n_positions:]
                labels = labels[:, -model.module.planner.encoder.config.n_positions:]
                weights = weights[:, -model.module.planner.encoder.config.n_positions:]
            input_ids = torch.clamp(input_ids, min=0,
                                    max=model.module.planner.encoder.config.vocab_size - 1)
            optimizer.zero_grad()
            with torch.no_grad():
                pred_future, _, _ = model.module.planner(prompt_ids, prompt_mask)

            logits = model.module(input_ids, pred_future)
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            global_step += 1

            if is_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "epoch": epoch + 1
                }, step=global_step)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "epoch": f"{epoch + 1}/{num_epochs}",
                    "step": f"{global_step}"
                })
        # --- Epoch End Validation Evaluation ---
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        avg_val_loss = run_validation(
            model.module, val_dataloader, device
        )
        model.module.train()
        model.module.planner.eval()

        if is_main_process:
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "epoch": epoch + 1
            }, step=global_step)

            print(f"\n--- Epoch {epoch+1} Complete ---")
            print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}\n")

            print(f"Epoch {epoch+1}/{num_epochs} complete. Saving checkpoint...")
            checkpoint_path = f"{outdir}/checkpoint-{global_step}.pt"
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}\n")
        dist.barrier()

    if is_main_process:
        print("Training Complete. Saving writer model components...")
        torch.save(model.module.state_dict(), f"{outdir}/writer_model.pt")

        print("Training finished!")
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    train(dataset_name="tinystories")
