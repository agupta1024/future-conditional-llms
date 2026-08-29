"""Writer Model Training Script for TinyStories, Treasure hunt Datasets.
DDP enabled."""

import gc
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import set_seed

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_th import get_dataloaders as get_th_dataloaders
from ..dataset.prepare_dataset_ts import get_dataloaders as get_ts_dataloaders

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

def train(dataset_name):
    # pylint: disable=too-many-locals, too-many-statements
    """Main training loop for the writer model."""
    torch.cuda.empty_cache()
    gc.collect()
    local_rank = setup_ddp()
    is_main_process = local_rank == 0
    device = torch.device(f"cuda:{local_rank}")
    print(f"Training on {device}")
    set_seed(42)

    base_working_model = 'gpt2_512'
    ds_name = dataset_name
    stage = "writer"

    dataset_config = get_dataset_config(name=ds_name)
    train_path = dataset_config.get("train_path", "")
    print(f"Loading {ds_name} dataset from {train_path}...")
    validation_path = dataset_config.get("val_path", "")
    batch_size = dataset_config.get("batch_size", 32)
    print("Preparing Datasets...")
    if "blocksworld" in ds_name:
        print("ERROR: Blocksworld dataset is not supported in this training script.")
        print("Please use the 'train_bw_writer.py' script for Blocksworld datasets.")
        return
    if ds_name == "treasure_hunt":
        dataloader_cls = get_th_dataloaders
    else:
        dataloader_cls = get_ts_dataloaders

    if ds_name != "tinystories":
        train_loader, eval_loader, _, train_sampler = dataloader_cls(
            train_path=train_path,
            eval_path=validation_path,
            batch_size=batch_size,
            tokenizer_path=dataset_config.get("tokenizer_path", ""),
            is_ddp=True
        )
    else:
        train_loader, eval_loader, _, train_sampler = dataloader_cls(
            dataset_name="skeskinen/TinyStories-Instruct-hf",
            batch_size=batch_size,
            is_ddp=True,
            max_length=512,
        )

    _, model = get_model_and_tokenizer(
        working_model=base_working_model,
        max_seq_length=512,
        load_scratch=True,
        dataset_name=ds_name,
        load_stage=stage,
        custom_ar=True,
        film=True,
        overwrite_planner=True,
        vocab_size=dataset_config.get("vocab_size", 50257),
        tokenizer_path=dataset_config.get("tokenizer_path", "")
    )
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    learning_rate = 1e-4
    num_epochs = 50

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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "epoch": f"{epoch + 1}/{num_epochs}",
                    "step": f"{global_step}"
                })
        # --- Epoch End Validation Evaluation ---
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = run_validation(
            model.module, eval_loader, device
        )
        model.module.train()
        model.module.planner.eval()

        if is_main_process:
            print(f"\n--- Epoch {epoch+1} Complete ---")
            print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}\n")

            print(f"Epoch {epoch+1}/{num_epochs} complete. Saving checkpoint...")
            checkpoint_path = f"{outdir}/checkpoint_{global_step}.pt"
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}\n")
        dist.barrier()

    if is_main_process:
        print("Training Complete. Saving writer model components...")
        torch.save(model.module.state_dict(), f"{outdir}/writer_model.pt")

        print("Training finished!")
    cleanup_ddp()

if __name__ == "__main__":
    train(dataset_name="treasure_hunt")
