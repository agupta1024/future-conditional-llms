"""Writer Model Training Script for Blocksworld Datasets."""

import os

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_ds_goal_drop import get_dataloaders as get_bw_goal_drop_dataloaders

def train_dynamic_e2e(objective=""):
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Train blocksworld Writer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting End-to-End Continuous Plan Training on {device}...")
    num_epochs=25
    learning_rate=1e-4

    base_working_model = 'gpt2_1024'
    ds_name = "blocksworld"
    stage = "writer"

    dataset_config = get_dataset_config(name=ds_name)
    train_path = dataset_config.get("train_path", "")
    print(f"Loading {ds_name} dataset from {train_path}...")
    validation_path = dataset_config.get("val_path", "")
    batch_size = dataset_config.get("batch_size", 32)
    print("Preparing Datasets...")
    if objective == "_goal_drop":
        dataloader_cls = get_bw_goal_drop_dataloaders
    else:
        dataloader_cls = get_bw_dataloaders
    train_loader, eval_loader, _, _ = dataloader_cls(
        train_path=train_path,
        eval_path=validation_path,
        batch_size=batch_size,
        tokenizer_path=dataset_config.get("tokenizer_path", ""),
        is_ddp=False
    )
    _, model = get_model_and_tokenizer(
        working_model=base_working_model,
        max_seq_length=1024,
        load_scratch=True,
        dataset_name=ds_name,
        load_stage=stage,
        custom_ar=True,
        film=True,
        vocab_size=dataset_config.get("vocab_size", 100),
        tokenizer_path=dataset_config.get("tokenizer_path", ""),
    )
    model.to(device)

    outdir = f'./{ds_name}_{stage}_static_model_{base_working_model}{objective}'
    os.makedirs(outdir, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        model.planner.encoder.eval()
        total_train_loss = 0

        print(f"\n=== Epoch {epoch + 1} / {num_epochs} ===")

        for _, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_weights = batch["loss_weights"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            update_mask = batch["update_mask"].to(device)
            film_mask = batch["loss_weights"].to(device)

            if input_ids.size(1) > model.config.n_positions:
                input_ids = input_ids[:, :model.config.n_positions]
                labels = labels[:, :model.config.n_positions]
                loss_weights = loss_weights[:, :model.config.n_positions]
                update_mask = update_mask[:, :model.config.n_positions]

            optimizer.zero_grad()
            b_size, seq_len = input_ids.shape
            p_0 = model.planner.get_initial_plan(prompt_ids, prompt_mask)
            p_seq_tensor = p_0.expand(-1, seq_len, -1)

            film_mask = torch.zeros_like(input_ids, dtype=torch.float, device=device)
            for i in range(b_size):
                prompt_len = int(prompt_mask[i].sum().item())
                film_mask[i, prompt_len - 1:] = 1.0

            logits = model(
                input_ids=input_ids,
                latent_plan=p_seq_tensor,
                film_mask=film_mask
            )

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()

            loss_per_token = loss_fct(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1)
            )
            flat_weights = shift_weights.view(-1)
            flat_labels = shift_labels.view(-1)
            valid_mask = (flat_labels != -100).float()
            effective_weights = flat_weights * valid_mask

            final_loss = ((loss_per_token * effective_weights).sum() /
                          (effective_weights.sum() + 1e-8))
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_train_loss += final_loss.item()
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
                loss_weights = batch["loss_weights"].to(device)
                prompt_ids = batch["prompt_ids"].to(device)
                prompt_mask = batch["prompt_mask"].to(device)
                update_mask = batch["update_mask"].to(device)

                b_size, seq_len = input_ids.shape

                p_0 = model.planner.get_initial_plan(prompt_ids, prompt_mask)
                traj_hidden = model.planner.encoder(input_ids, return_hidden_states=True)

                p_seq = []
                p_curr = p_0

                for t in range(seq_len):
                    has_update = update_mask[:, t]
                    if has_update.any():
                        act_emb = traj_hidden[:, t, :]
                        p_next = model.planner.step_plan(act_emb, p_curr)
                        mask_expanded = has_update.bool().view(b_size, 1, 1).expand_as(p_curr)
                        p_curr = torch.where(mask_expanded, p_next, p_curr)
                    p_seq.append(p_curr)

                p_seq_tensor = torch.cat(p_seq, dim=1)
                film_mask = torch.zeros_like(input_ids, dtype=torch.float)
                for i in range(b_size):
                    prompt_len = int(prompt_mask[i].sum().item())
                    film_mask[i, prompt_len - 1:] = 1.0

                logits = model(
                    input_ids=input_ids,
                    latent_plan=p_seq_tensor,
                    film_mask=film_mask
                )

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_weights = loss_weights[..., 1:].contiguous()

                loss_per_token = loss_fct(
                    shift_logits.view(-1, model.config.vocab_size),
                    shift_labels.view(-1)
                )
                flat_weights = shift_weights.view(-1)
                flat_labels = shift_labels.view(-1)
                valid_mask = (flat_labels != -100).float()
                effective_weights = flat_weights * valid_mask

                final_loss = ((loss_per_token * effective_weights).sum() /
                              (effective_weights.sum() + 1e-8))
                total_eval_loss += final_loss.item()

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {total_train_loss / len(train_loader):.4f}")
        print(f"  Eval Loss:  {total_eval_loss / len(eval_loader):.4f}")

    print(f"\nSaving Dynamic E2E Model to {outdir}...")
    torch.save(model.state_dict(), f"{outdir}/writer_model.pt")
    print("Training Complete!")

if __name__ == "__main__":
    train_dynamic_e2e(objective="")
    # train_dynamic_e2e(objective="_goal_drop")
