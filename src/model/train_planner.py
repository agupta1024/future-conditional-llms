"""Training script for the Latent Planner model using VICReg loss."""

import os
import gc
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import set_seed
import wandb

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_th import get_dataloaders as get_th_dataloaders
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_dataset_ts import get_dataloaders as get_ts_dataloaders
from .utils import init_wandb

# --- VICReg Math Helpers ---
def off_diagonal(x):
    """Returns a flattened view of the off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(pred_future, true_future, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0):
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """
    Calculates the VICReg loss between predicted and true latent representations.
    Default coefficients (25, 25, 1) are standard from the original paper.
    """
    batch_size, hidden_dim = pred_future.shape

    # 1. Invariance Loss (MSE)
    sim_loss = F.mse_loss(pred_future, true_future)

    # 2. Variance Loss (Hinge loss on standard deviation)
    # Adding epsilon (1e-04) for numerical stability in sqrt
    std_pred = torch.sqrt(pred_future.var(dim=0) + 1e-04)
    std_true = torch.sqrt(true_future.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(gamma - std_pred)) / 2 + \
               torch.mean(F.relu(gamma - std_true)) / 2

    # 3. Covariance Loss
    # Center the representations
    pred_future = pred_future - pred_future.mean(dim=0)
    true_future = true_future - true_future.mean(dim=0)

    cov_pred = (pred_future.T @ pred_future) / (batch_size - 1)
    cov_true = (true_future.T @ true_future) / (batch_size - 1)

    cov_loss = off_diagonal(cov_pred).pow_(2).sum().div(hidden_dim) + \
               off_diagonal(cov_true).pow_(2).sum().div(hidden_dim)

    loss = (sim_coeff * sim_loss) + (std_coeff * std_loss) + (cov_coeff * cov_loss)
    return loss, sim_loss, std_loss, cov_loss

def save_checkpoint(model, step, save_dir):
    """Save the model checkpoint at the specified step."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint-{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# --- Training Loop ---
def train(dataset_name="tinystories"):
    # pylint: disable=too-many-locals,too-many-statements
    """Main training loop for the Latent Planner model."""
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    set_seed(42)

    base_working_model = 'gpt2_512_512'
    ds_name = dataset_name
    stage = "planner"
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
    dataset_config = get_dataset_config(name=ds_name, stage=stage)
    train_path = dataset_config.get("train_path", "")
    print(f"Loading {ds_name} dataset from {train_path}...")
    validation_path = dataset_config.get("val_path", "")
    batch_size = dataset_config.get("batch_size", 16)

    print("Preparing Datasets...")
    learning_rate = 5e-5
    num_epochs = 20
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
    wandb_run_name= f"{stage}-{ds_name}-{base_working_model}"
    init_wandb(
        model_name=base_working_model,
        stage=stage,
        ds_name=ds_name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        ds_size=len(train_dataloader.dataset),
        run_name=wandb_run_name,
    )

    outdir = f'./{ds_name}_{stage}_model_{base_working_model}'
    os.makedirs(outdir, exist_ok=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen Foundation Parameters: {frozen_params:,}")
    print(f"Trainable Planning Parameters: {trainable_params:,}")
    optimizer = torch.optim.AdamW(model.future_predictor.parameters(),
                                  lr=learning_rate, weight_decay=1e-4)
    global_step = 0
    eval_steps = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for _, batch in enumerate(progress_bar):
            prompt_ids = batch['prompt_ids'].to(device)
            prompt_mask = batch['prompt_mask'].to(device)
            input_ids = batch.get('input_ids', prompt_ids).to(device)
            attention_mask = batch.get('attention_mask', prompt_mask).to(device)
            if input_ids.size(1) > model.encoder.config.n_positions:
                input_ids = input_ids[:, -model.encoder.config.n_positions:]
                attention_mask = attention_mask[:, -model.encoder.config.n_positions:]
            optimizer.zero_grad()

            pred_future, true_future, _ = model(prompt_ids, prompt_mask, future_ids=input_ids,
                                                future_mask=attention_mask)

            loss, sim, std, cov = vicreg_loss(pred_future, true_future)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            wandb.log({
                "train/future_predict_loss": loss.item(),
                "train/sim_loss": sim.item(),
                "train/std_loss": std.item(),
                "train/cov_loss": cov.item(),
            }, step=global_step)

            if global_step % eval_steps == 0 and global_step > 0:
                print(f"\nRunning Validation at Step {global_step}...")
                avg_pos_sim, avg_neg_sim, avg_loss = run_validation(model, eval_dataloader, device)
                model.train()
                wandb.log({
                    "val/positive_similarity": avg_pos_sim,
                    "val/negative_similarity": avg_neg_sim,
                    "val/loss": avg_loss,
                }, step=global_step)

            progress_bar.set_postfix({
                "ENC_loss": f"{loss.item():.3f}", 
            })
            global_step += 1

        if (epoch + 1) % 5   == 0:
            print(f"Epoch {epoch+1}/{num_epochs} complete. Saving checkpoint...")
            save_checkpoint(model, global_step, outdir)

    print("Training Complete. Saving Future Encoder components...")
    torch.save(model.state_dict(), f"{outdir}/planner_model.pt")

def run_validation(model, val_dataloader, device):
    # pylint: disable=too-many-locals
    """Run validation on the model using the provided dataloader."""
    model.eval()
    total_positive_sim = 0.0
    total_negative_sim = 0.0
    batches_processed = 0
    total_loss = 0.0

    print("Running Evaluation...")
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['prompt_ids'].to(device)
            attention_mask = batch['prompt_mask'].to(device)
            future_ids = batch.get('input_ids', input_ids).to(device)
            future_mask = batch.get('attention_mask', attention_mask).to(device)

            batch_size = input_ids.size(0)

            pred_future, true_future, _ = model(input_ids, attention_mask,
                                                future_ids=future_ids, future_mask=future_mask)
            loss, _, _, _ = vicreg_loss(pred_future, true_future)
            total_loss += loss.item()
            pred_future_norm = F.normalize(pred_future, p=2, dim=1)
            true_future_norm = F.normalize(true_future, p=2, dim=1)

            similarity_matrix = torch.matmul(pred_future_norm, true_future_norm.T)
            positive_sim = torch.diag(similarity_matrix).mean().item()

            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            negative_sim = similarity_matrix[mask].mean().item()

            total_positive_sim += positive_sim
            total_negative_sim += negative_sim
            batches_processed += 1

    avg_pos = total_positive_sim / batches_processed
    avg_neg = total_negative_sim / batches_processed
    avg_loss = total_loss / batches_processed
    print(f"Semantic Gap (Higher is better):                {avg_pos - avg_neg:.4f}")

    return avg_pos, avg_neg, avg_loss

if __name__ == "__main__":
    train(dataset_name="tinystories")
