"""Training script for the Latent Planner model using VICReg loss."""

import os
import gc
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import set_seed

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_th import get_dataloaders as get_th_dataloaders
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_dataset_ts import get_dataloaders as get_ts_dataloaders

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
    sim_loss = F.mse_loss(pred_future, true_future)

    std_pred = torch.sqrt(pred_future.var(dim=0) + 1e-04)
    std_true = torch.sqrt(true_future.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(gamma - std_pred)) / 2 + \
               torch.mean(F.relu(gamma - std_true)) / 2

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

def train(dataset_name):
    # pylint: disable=too-many-locals,too-many-statements
    """Main training loop for the Latent Planner model."""
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    set_seed(42)

    base_working_model = 'gpt2_512'
    ds_name = dataset_name
    stage = "planner"

    dataset_config = get_dataset_config(name=ds_name)
    train_path = dataset_config.get("train_path", "")
    print(f"Loading {ds_name} dataset from {train_path}...")
    validation_path = dataset_config.get("val_path", "")
    batch_size = dataset_config.get("batch_size", 16)

    if ds_name == "treasure_hunt":
        dataloader_cls = get_th_dataloaders
    elif ds_name == "blocksworld":
        dataloader_cls = get_bw_dataloaders
    else:
        dataloader_cls = get_ts_dataloaders

    if ds_name != "tinystories":
        train_dataloader, eval_dataloader, _, _ = dataloader_cls(
            train_path=train_path,
            eval_path=validation_path,
            batch_size=batch_size,
            tokenizer_path=dataset_config.get("tokenizer_path", ""),
        )
    else:
        train_dataloader, eval_dataloader, _, _ = dataloader_cls(
            dataset_name="skeskinen/TinyStories-Instruct-hf",
            batch_size=batch_size,
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
        vocab_size=dataset_config.get("vocab_size", 50257),
        tokenizer_path=dataset_config.get("tokenizer_path", "")
    )
    model.to(device)

    learning_rate = 5e-5
    num_epochs = 20

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

            loss, _, _, _ = vicreg_loss(pred_future, true_future)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if global_step % eval_steps == 0 and global_step > 0:
                print(f"\nRunning Validation at Step {global_step}...")
                _, _, avg_loss = run_validation(model, eval_dataloader, device)
                print(f"Validation Loss at Step {global_step}: {avg_loss:.4f}")
                model.train()

            progress_bar.set_postfix({
                "ENC_loss": f"{loss.item():.3f}"
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
    train(dataset_name="treasure_hunt")
