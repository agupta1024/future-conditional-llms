""" Blocksworld Ablation Script"""

import json
import torch
from transformers import set_seed

from .blocksworld_simulator import BWSimulator
from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer

def evaluate_model(mode, prompts, tokenizer, model, device, num_samples, comma_id, eos_id):
    # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-statements
    """ Evaluates the model's performance for a given mode."""
    successes = 0
    legal_trajectories = 0
    results = {}
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.size(1)
        if BWSimulator is not None:
            init_str = prompt.split("[GOAL]")[0].replace("[INIT]", "").strip()
            goal_str = prompt.split("[GOAL]")[1].strip()
            goal_conditions = {c.strip() for c in goal_str.split(",") if c.strip()}

        with torch.no_grad():
            prompt_mask = torch.ones_like(input_ids).to(device)
            p_curr = model.planner.get_initial_plan(input_ids, prompt_mask)

        if "Random" in mode:
            p_curr = torch.randn_like(p_curr) * p_curr.std() + p_curr.mean()

        plan_history = p_curr.expand(-1, prompt_len, -1).clone()
        generated_ids = input_ids.clone()

        for _ in range(50):
            seq_len = generated_ids.size(1)
            film_mask = (torch.arange(seq_len, device=device) >=
                         (prompt_len - 1)).float().unsqueeze(0)

            with torch.no_grad():
                logits = model(generated_ids, latent_plan=plan_history, film_mask=film_mask)

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            if next_token.item() == eos_id:
                break
            if "Static" in mode or "Random" in mode:
                plan_history = torch.cat((plan_history, p_curr), dim=1)
                continue

            if next_token.item() == comma_id:
                with torch.no_grad():
                    enc_out = model.planner.encoder(generated_ids, return_hidden_states=True)
                    act_emb = enc_out[:, -1, :]
                    p_curr = model.planner.step_plan(act_emb, p_curr)
            plan_history = torch.cat((plan_history, p_curr), dim=1)

        gen_text = tokenizer.decode(generated_ids[0][prompt_len:])
        for token in ["[DONE]", "[PAD]", "<|endoftext|>"]:
            if token in gen_text:
                gen_text = gen_text.split(token)[0]

        actions = [act.strip() for act in gen_text.split(",") if act.strip()]
        if BWSimulator is not None:
            sim = BWSimulator(init_str)
            is_legal = True
            for action in actions:
                if not sim.apply_action(action):
                    is_legal = False
                    break
            if is_legal:
                legal_trajectories += 1
                if goal_conditions.issubset(sim.state):
                    successes += 1
        else:
            if "stack" in gen_text or "putdown" in gen_text or "pickup" in gen_text:
                successes += 1

    results = {"success_rate":(successes / num_samples) * 100,
               "legal_action_rate":(legal_trajectories / num_samples) * 100}
    return results

def run_zero_shot_ablations():
    # pylint: disable=too-many-locals
    """
    Tackles Ablations for the FiLM-based Planner in Blocksworld. Evaluates the following:
    1. Dynamic FiLM (Ours)
    2. Static FiLM
    3. Random FiLM
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading environment on {device}...")
    set_seed(42)
    num_samples=100

    ds_name = "blocksworld"
    dataset_config = get_dataset_config(name=ds_name)
    eval_data_path = dataset_config.get("val_path", "")

    model_config = {
        'max_seq_length': 1024,
        'load_scratch': False,
        'dataset_name': ds_name,
        'custom_ar': True,
        'film': True,
        'vocab_size': dataset_config.get("vocab_size", 100),
        'tokenizer_path': dataset_config.get("tokenizer_path", ""),
    }

    model_config['working_model'] = 'gpt2_1024'
    model_config['load_stage'] = 'writer'
    tokenizer, dynamic_model = get_model_and_tokenizer(**model_config)
    dynamic_model.to(device)
    dynamic_model.eval()

    static_model_path = "./blocksworld_writer_static_model_gpt2_1024/writer_model.pt"
    model_config['writer_model_path'] = static_model_path
    _, static_model = get_model_and_tokenizer(**model_config)
    static_model.to(device)
    static_model.eval()
    comma_id = tokenizer.convert_tokens_to_ids(",")
    eos_id = tokenizer.convert_tokens_to_ids("[DONE]")

    prompts = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"] + " ")
            if len(prompts) >= num_samples:
                break

    modes = ["Dynamic (Ours)", "Static FiLM", "Random FiLM"]
    results = {m: 0 for m in modes}

    print("\n=== RUNNING ZERO-SHOT STRUCTURAL ABLATIONS ===")

    for mode in modes:
        print(f"\n--- Evaluating: {mode} ---")
        if mode == "Dynamic (Ours)" or "Random" in mode:
            results[mode] = evaluate_model(mode, prompts, tokenizer, dynamic_model,
                                           device, num_samples, comma_id, eos_id)
        elif mode == "Static FiLM":
            results[mode] = evaluate_model(mode, prompts, tokenizer, static_model,
                                           device, num_samples, comma_id, eos_id)

    print("\n=== ABLATION RESULTS ===")
    for mode in modes:
        print(f"{mode}: {results[mode]}%")

if __name__ == "__main__":
    run_zero_shot_ablations()
