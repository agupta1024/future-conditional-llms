"""Writer Model Training Script for Blocksworld Datasets."""

import os
import re

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_ds_goal_drop import get_dataloaders as get_bw_goal_drop_dataloaders
from ..evaluation.blocksworld_simulator import BWSimulator

def verify_blocksworld_trajectory(prompt: str, generated_text: str) -> float:
    """
    Main RL Verifier function.
    
    Returns:
        1.0: If trajectory reaches the goal without violating physics/preconditions.
        0.0: If an action is invalid, impossible, or fails to reach the final goal.
    """
    # 1. Parse [INIT] and [GOAL] segments from prompt
    if BWSimulator is not None:
        init_str = prompt.split("[GOAL]")[0].replace("[INIT]", "").strip()
        goal_str = prompt.split("[GOAL]")[1].strip()

    if not init_str or not goal_str:
        return 0.0

    # 2. Truncate generated text at [DONE] if present
    done_idx = generated_text.find("[DONE]")
    if done_idx != -1:
        generated_text = generated_text[:done_idx]

    # Split generated actions by commas or newlines
    actions = [a.strip() for a in re.split(r'[,;\n]+', generated_text) if a.strip()]

    # 3. Simulate state transitions step-by-step
    if BWSimulator is not None:
        sim = BWSimulator(init_str)
        # print(f"\n[Actions]: {actions}")
        for action in actions:
            # print(f"Applying action: {action.strip()}")
            if not sim.apply_action(action):
                return 0.0

    # 4. Check if final state satisfies all goal predicates
    score = sim.score_goal(goal_str)
    total_success = int(score["success"])
    if total_success > 0:
        return 1.0

    return 0.0
    
def train_dynamic_e2e(objective=""):
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Train blocksworld Writer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting End-to-End Continuous Plan Training on {device}...")
    num_epochs=20
    learning_rate=1e-5

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
        is_ddp=False,
        padding="left",
    )

    tokenizer, model = get_model_and_tokenizer(
        working_model=base_working_model,
        max_seq_length=1024,
        load_scratch=False,
        dataset_name=ds_name,
        load_stage=stage,
        custom_ar=True,
        film=True,
        vocab_size=dataset_config.get("vocab_size", 100),
        tokenizer_path=dataset_config.get("tokenizer_path", ""),
    )
    model.to(device)

    outdir = f'./{ds_name}_{stage}_model_{base_working_model}{objective}_REINFORCE'
    os.makedirs(outdir, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    # Baseline tracker for variance reduction (moving average of rewards)
    running_baseline = 0.5
    baseline_alpha = 0.1  # Moving average momentum
    comma_id = tokenizer.encode(",")[0]
    done_token_id = tokenizer.encode("[DONE]")[0]

    for epoch in range(num_epochs):
        model.train()
        model.planner.encoder.eval()
        total_train_loss = 0
        total_reward = 0.0

        print(f"\n=== REINFORCE Epoch {epoch + 1} / {num_epochs} ===")
        pbar = tqdm(eval_loader, desc="RL Training")
        for batch_idx, batch in enumerate(pbar):
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            b_size = prompt_ids.size(0)

            # ---------------------------------------------------------------------
            # STAGE 1: Autoregressive Rollout (Self-Generation)
            # ---------------------------------------------------------------------
            model.eval()
            with torch.no_grad():
                gen_input_ids = prompt_ids.clone()
                p_0 = model.planner.get_initial_plan(prompt_ids, prompt_mask)
                p_curr = p_0
                
                # Track update triggers for generated tokens
                gen_update_mask = torch.zeros_like(prompt_ids, dtype=torch.bool)
                
                # Create a FiLM mask for the generated tokens
                attention_mask = (prompt_ids != tokenizer.pad_token_id).long()
                film_mask = torch.zeros_like(prompt_ids, dtype=torch.float, device=device)
                film_mask[:, -1] = 1.0
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
    
                max_gen_len = min(40, model.config.n_positions - prompt_ids.size(1))
                past_key_values = None
                current_input = prompt_ids
                curr_pos_ids = position_ids
                unfinished_sequences = torch.ones(b_size, dtype=torch.bool, device=device)
                for _ in range(max_gen_len):
                    curr_len = current_input.size(1)
                    p_seq_tensor = p_curr.expand(-1, curr_len, -1)

                    # 4. Get logits for next token prediction
                    logits,past_key_values = model(
                        input_ids=current_input,
                        attention_mask=attention_mask,
                        position_ids=curr_pos_ids,
                        latent_plan=p_seq_tensor,
                        film_mask=film_mask,
                        use_cache=True,
                        kv_caches=past_key_values
                    )
                    next_token_logits = logits[:, -1, :]
                    next_token_logits[:, tokenizer.pad_token_id] = -float('inf')
                    film_mask = torch.ones((b_size, 1), dtype=torch.float, device=device)
                    # 5. Sample next token using multinomial policy
                    top_k = 5
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    temperature = 0.15
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1) # [b_size, 1]
                    # --- FREEZE FINISHED SEQUENCES ---
                    # If a sequence already hit [DONE] in a previous step, force its 
                    # next token to just be a [PAD] so it stops generating garbage.
                    next_tokens = next_tokens * unfinished_sequences.unsqueeze(1).long() + \
                                tokenizer.pad_token_id * (~unfinished_sequences).unsqueeze(1).long()
                                
                    # Update which sequences are still active
                    has_hit_done = (next_tokens.squeeze(-1) == done_token_id)
                    unfinished_sequences = unfinished_sequences & ~has_hit_done
                    # 6. Check for trigger condition (e.g., token ID corresponding to ',')
                    is_trigger = (next_tokens.squeeze(-1) == comma_id)
                    
                    if is_trigger.any():
                        # 1. Compute hidden states of current generated sequence for planner
                        planner_attn_mask = (gen_input_ids != tokenizer.pad_token_id).long()
                        traj_hidden = model.planner.encoder(gen_input_ids,
                                                            attention_mask=planner_attn_mask,
                                                            return_hidden_states=True)
                        act_emb = traj_hidden[:, -1, :]
                        p_next = model.planner.step_plan(act_emb, p_curr)
                        mask_expanded = is_trigger.view(b_size, 1, 1).expand_as(p_curr)
                        p_curr = torch.where(mask_expanded, p_next, p_curr)

                    # 7. Append sampled tokens to sequence
                    gen_input_ids = torch.cat([gen_input_ids, next_tokens], dim=1)
                    gen_update_mask = torch.cat([gen_update_mask, is_trigger.unsqueeze(1)], dim=1)
                    current_input = next_tokens
                    curr_pos_ids = curr_pos_ids[:, -1:] + 1
                    position_ids = torch.cat([position_ids, curr_pos_ids], dim=1)
                    
                    # --- UPDATE ATTENTION MASK FOR KV CACHE ---
                    # Add a '1' to the attention mask for the newly generated token
                    new_mask = torch.ones((b_size, 1), dtype=torch.long, device=device)
                    attention_mask = torch.cat([attention_mask, new_mask], dim=1)
                    # Stop early if all sequences generated [DONE] token
                    if not unfinished_sequences.any():
                        break
                    # if (next_tokens.squeeze(-1) == done_token_id).all():
                    #     break
        
            # ---------------------------------------------------------------------
            # STAGE 2: Environment Evaluation (Rewards)
            # ---------------------------------------------------------------------
            rewards = []
                
            for i in range(b_size):
                p_text = tokenizer.decode(prompt_ids[i])
                g_text = tokenizer.decode(gen_input_ids[i][prompt_ids.size(1):])
                # print(f"Prompt: {p_text}")
                # print(f"Generated: {g_text}")
                # Environment verifier returns 1.0 for goal reach, 0.0 otherwise
                r = verify_blocksworld_trajectory(p_text, g_text)
                print(f"PROMPT: {repr(p_text)}")
                print(f"GENERATED: {repr(g_text)}")
                print(f"Reward for sample {i}: {r}")
                breakpoint()
                rewards.append(r)
                
            breakpoint()
            reward_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
            batch_mean_reward = reward_tensor.mean().item()
            total_reward += batch_mean_reward

            # Compute Advantage using moving average baseline: A = R - b
            advantages = reward_tensor - running_baseline
            running_baseline = (1 - baseline_alpha) * running_baseline + baseline_alpha * batch_mean_reward
            current_running_avg = total_reward / (batch_idx + 1)
            pbar.set_postfix({
                "Batch Succ": f"{batch_mean_reward * 100:.1f}%",
                "Avg Succ": f"{current_running_avg * 100:.1f}%"
            })
            # ---------------------------------------------------------------------
            # STAGE 3: Policy Gradient Optimization (Re-evaluate Forward Pass)
            # ---------------------------------------------------------------------
            model.train()
            optimizer.zero_grad()

            b_size, gen_seq_len = gen_input_ids.shape
            p_0 = model.planner.get_initial_plan(prompt_ids, prompt_mask)
            traj_hidden = model.planner.encoder(gen_input_ids, return_hidden_states=True)

            # Reconstruct plan sequence with active gradients enabled
            p_seq = []
            p_curr = p_0

            for t in range(gen_seq_len):
                has_update = gen_update_mask[:, t]
                if has_update.any():
                    act_emb = traj_hidden[:, t, :]
                    p_next = model.planner.step_plan(act_emb, p_curr)
                    mask_expanded = has_update.bool().view(b_size, 1, 1).expand_as(p_curr)
                    p_curr = torch.where(mask_expanded, p_next, p_curr)

                p_seq.append(p_curr)
            p_seq_tensor = torch.cat(p_seq, dim=1)

            film_mask = torch.zeros_like(gen_input_ids, dtype=torch.float, device=device)
            for i in range(b_size):
                prompt_len = int(prompt_mask[i].sum().item())
                film_mask[i, prompt_len - 1:] = 1.0

            # Model forward pass on generated trajectory
            logits = model(
                input_ids=gen_input_ids,
                latent_plan=p_seq_tensor,
                film_mask=film_mask
            )

            # Compute log probabilities of generated non-prompt actions
            shift_logits = logits[..., :-1, :].contiguous()
            shift_tokens = gen_input_ids[..., 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            sampled_log_probs = log_probs.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)

            # Create mask so REINFORCE loss only applies to generated tokens (not prompt)
            action_mask = torch.zeros_like(shift_tokens, dtype=torch.float, device=device)
            for i in range(b_size):
                p_len = int(prompt_mask[i].sum().item())
                action_mask[i, p_len - 1:] = 1.0

            # REINFORCE Loss: - sum( log_pi(a_t | s_t) * Advantage )
            # Reshape advantages to broadcast across sequence dimension [b_size, 1]
            advantage_targets = advantages.unsqueeze(1).expand_as(sampled_log_probs)
            
            policy_loss = - (sampled_log_probs * advantage_targets * action_mask).sum() / (action_mask.sum() + 1e-8)

            # Backward and step
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += policy_loss.item()
            
        epoch_success_rate = total_reward / len(train_loader)
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  -> Final Success Rate: {epoch_success_rate * 100:.2f}%")
        print(f"  -> Average Policy Loss: {total_train_loss / len(train_loader):.4f}")
        print(f"  -> Current Baseline: {running_baseline:.4f}")

        
            
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1 or True:
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
