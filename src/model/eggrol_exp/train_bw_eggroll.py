import os
import re
import math
import torch
import numpy as np
from tqdm import tqdm
from .model_bw import inject_eggroll_adapters, EggrollAdapter, remove_eggroll_adapters_and_save # Assuming the code above is in model.py
from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.prepare_dataset_bw import get_dataloaders as get_bw_dataloaders
from ..dataset.prepare_ds_goal_drop import get_dataloaders as get_bw_goal_drop_dataloaders
from ..evaluation.blocksworld_simulator import BWSimulator

def env_verifier(prompt: str, generated_text: str) -> float:
    """
    Main RL Verifier function.
    
    Returns:
        1.0: If trajectory reaches the goal without violating physics/preconditions.
        0.0: If an action is invalid, impossible, or fails to reach the final goal.
    """
    # 1. Parse [INIT] and [GOAL] segments from prompt
    if BWSimulator is not None:
        init_str = prompt.split("[GOAL]")[0].replace("[INIT]", "").strip()
        init_str = init_str.replace("[PAD]", "").strip()
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

# --- EGGROLL TRAINING ENGINE ---
def train_eggroll(objective):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting End-to-End Continuous Plan Training on {device}...")
    
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
    hidden_dim = model.config.hidden_size
    NOISE_SCALE = 0.1 / math.sqrt(hidden_dim)
    model.to(device)
    
    outdir = f'./{ds_name}_{stage}_model_{base_working_model}{objective}_EGGROLL'
    os.makedirs(outdir, exist_ok=True)
        
    # 1. Inject Evolutionary Adapters
    model = inject_eggroll_adapters(model, rank=16)
    model.eval() # Model stays in eval mode permanently (no dropout, no batchnorm shifts)
    
    # EGGROLL Hyperparameters
    POPULATION_SIZE = 16   # Number of mutations to try per batch
    LEARNING_RATE = 0.0001   # How fast the central weights move toward successful mutations
    
    # Extract all Eggroll layers to orchestrate them centrally
    eggroll_layers = [m for m in model.modules() if isinstance(m, EggrollAdapter)]
    print(f"Injected {len(eggroll_layers)} EGGROLL adapters.")
    print("Starting EGGROLL Evolution...")
    for layer in eggroll_layers:
        layer.delta_W.data.zero_()
    print("Initialized all delta_W parameters to 0.0.")
    for epoch in range(1):
        print(f"\n=== EGGROLL Epoch {epoch + 1} ===")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Evolving")):
            input_ids = batch["prompt_ids"].to(device)
            attention_mask = batch["prompt_mask"].to(device)
            
            # Generate N distinct seeds for the population
            # Using standard python ints to ensure cross-platform reproducibility
            population_seeds = [None] + [int(torch.randint(0, 2**32 - 1, (1,)).item()) for _ in range(POPULATION_SIZE - 1)]
            population_rewards = []

            # ==========================================
            # PHASE 1: EVALUATE THE POPULATION
            # ==========================================
            for seed in population_seeds:
                if seed is None:
                    # Unmutated base model pass
                    for layer in eggroll_layers:
                        layer.clear_noise()
                else:
                    # Mutated model pass
                    # Create a PRNG generator strictly for this population member
                    gen = torch.Generator(device=device).manual_seed(seed)
                    
                    # Mutate all layers synchronously using the member's seed
                    for layer in eggroll_layers:
                        layer.inject_noise(generator=gen, noise_scale=NOISE_SCALE)
                
                # 2. Pure Inference Generation
                # We use Greedy decoding (temp=0.0) because the exploration 
                # is happening in the WEIGHTS, not the tokens!
                with torch.no_grad():
                    generated_ids = model.generate_batched(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=40,
                        do_sample=False, # Pure deterministic Blocksworld logic
                        pad_token_id=tokenizer.pad_token_id,
                        comma_id=tokenizer.convert_tokens_to_ids(","),
                        eos_token_id=tokenizer.convert_tokens_to_ids("[DONE]")
                    )
                
                # 3. Score the batch
                batch_reward = 0
                for i in range(input_ids.size(0)):
                    prompt_len = input_ids.size(1)
                    p_text = tokenizer.decode(input_ids[i])
                    g_text = tokenizer.decode(generated_ids[i, prompt_len:])
                    # success = env_verifier(p_text, g_text)
                    # if success > 0:
                    #     print(f"\n[Success] Prompt: {p_text} | Generated: {g_text}")
                    # else:
                    #     print(f"\n[Failure] Prompt: {p_text} | Generated: {g_text}")
                    # batch_reward += success
                    batch_reward += env_verifier(p_text, g_text)
                
                mean_member_reward = batch_reward / input_ids.size(0)
                population_rewards.append(mean_member_reward)
                # print(f"Member Seed: {seed} | Reward: {batch_reward:.1f} | Mean: {mean_member_reward:.3f}")
            
            # ==========================================
            # PHASE 2: EVOLUTIONARY UPDATE
            # ==========================================
            # 1. Normalize fitness scores
            pop_rewards_np = np.array(population_rewards)
            base_model_reward = population_rewards[0] # Seed None is Member 0
            max_reward = np.max(pop_rewards_np)
            reward_mean = np.mean(pop_rewards_np)
            reward_std = np.std(pop_rewards_np) + 1e-8
            
            # Fitness F_i is centered at 0. Bad runs are negative, good runs are positive.
            fitness_scores = (pop_rewards_np - reward_mean) / reward_std
            
            # 2. Update Central Weights
            for seed, fitness in zip(population_seeds, fitness_scores):
                # The ES Update Rule: Theta = Theta + LR * (Fitness / (Sigma * N)) * Noise
                # We calculate the step size once per population member
                if seed is None:
                    continue
                step_size = LEARNING_RATE * (fitness / (NOISE_SCALE * POPULATION_SIZE))
                # 3. Clip step size to prevent huge weight jumps
                step_size = max(min(step_size, 0.005), -0.005)
                # Reset the PRNG generator using the EXACT SAME SEED
                gen = torch.Generator(device=device).manual_seed(seed)
                
                for layer in eggroll_layers:
                    # Regenerates the same noise tensor and accumulates the update
                    layer.accumulate_update(generator=gen, step_size=step_size)
            
            # 3. Reset for next batch
            for layer in eggroll_layers:
                layer.clear_noise()
                
            max_batch_reward = np.max(pop_rewards_np)
            avg_delta_norm = np.mean([layer.get_delta_norm() for layer in eggroll_layers])
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Max Batch Succ: {max_batch_reward * 100:.1f}% | Mean: {reward_mean * 100:.1f}%\n")
                print(
                    f"Batch {batch_idx:03d} | "
                    f"Base Model: {base_model_reward * 100:.1f}% | "
                    f"Pop Mean: {reward_mean * 100:.1f}% | "
                    f"Pop Max: {max_reward * 100:.1f}% | "
                    f"Avg Delta Norm: {avg_delta_norm:.4f}"
                )
                
            # print(f"Batch {batch_idx} | Max Reward: {np.max(pop_rewards_np)} | Mean: {reward_mean}")
            
    remove_eggroll_adapters_and_save(model, outdir)
    
if __name__ == "__main__":
    train_eggroll(objective="")