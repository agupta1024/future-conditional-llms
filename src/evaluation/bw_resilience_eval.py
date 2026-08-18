""" Blocksworld Resilience Evaluation Script over k Perturbations """

import gc
import json
import torch
from transformers import set_seed

from .blocksworld_simulator import BWSimulator
from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.generate_bw_sub import get_valid_actions

def run_k_perturbation_benchmark():
    # pylint: disable=too-many-locals,too-many-statements
    """Run the k-perturbation resilience benchmark."""
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eval suite on {device}")
    set_seed(42)
    print(f"Loading Models on {device}...")

    base_working_model = 'gpt2_1024'
    ds_name = "blocksworld_sub"
    stage = "writer"

    dataset_config = get_dataset_config(name=ds_name)
    eval_data_path = dataset_config.get("val_path", "")

    model_config = {
        'working_model': base_working_model,
        'max_seq_length': 1024,
        'load_scratch': False,
        'dataset_name': ds_name,
        'load_stage': stage,
        'custom_ar': True,
        'film': True,
        'vocab_size': dataset_config.get("vocab_size", 100),
        'tokenizer_path': dataset_config.get("tokenizer_path", ""),
    }

    tokenizer, dynamic_model = get_model_and_tokenizer(**model_config)
    dynamic_model.to(device)
    dynamic_model.eval()

    base_working_model = 'gpt2_1024-l'
    model_config['working_model'] = base_working_model
    model_config['load_stage'] = 'base'
    _, baseline = get_model_and_tokenizer(**model_config)
    baseline.to(device)
    baseline.eval()

    num_samples=100
    comma_id = tokenizer.convert_tokens_to_ids(",")
    eos_id = tokenizer.convert_tokens_to_ids("[DONE]")

    perturbation_levels = [0, 1, 2] # Test with 0, 1, 2, and 3 forced mistakes
    results = {"Dynamic": {k: 0 for k in perturbation_levels},
               "Baseline": {k: 0 for k in perturbation_levels}}

    prompts = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"] + " ")
            if len(prompts) >= num_samples:
                break

    print("\n=== RUNNING K-PERTURBATION RESILIENCE BENCHMARK ===")

    for k_mistakes in perturbation_levels:
        print(f"\n--- Testing with {k_mistakes} Forced Perturbations ---")

        dynamic_success = 0
        baseline_success = 0

        generated_trajectories = []
        for _, prompt in enumerate(prompts):
            init_str = prompt.split("[GOAL]")[0].replace("[INIT]", "").strip()
            goal_str = prompt.split("[GOAL]")[1].strip()
            goal_conditions = {c.strip() for c in goal_str.split(",") if c.strip()}

            def generate_trajectory(model, is_dynamic, init_str, prompt, goal_conditions,
                                    k_mistakes):
                # pylint: disable=too-many-arguments,too-many-positional-arguments
                sim = BWSimulator(init_str)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                prompt_len = input_ids.size(1)
                prompt_mask = torch.ones_like(input_ids)
                # print(f"\nEvaluating Prompt {idx+1}/{len(prompts)}: {prompt.strip()}")
                if is_dynamic:
                    p_curr = model.planner.get_initial_plan(input_ids, prompt_mask)
                    plan_history = p_curr.expand(-1, input_ids.size(1), -1).clone()

                actions_taken = []
                mistakes_injected = 0
                generated_ids = input_ids.clone()
                new_action_tokens = []
                for _ in range(50):
                    if is_dynamic:
                        seq_len = generated_ids.size(1)
                        film_mask = (torch.arange(seq_len,
                                                  device=input_ids.device) >= prompt_len
                                     ).float().unsqueeze(0).expand(generated_ids.size(0), -1)
                        with torch.no_grad():
                            logits = model(
                                input_ids=generated_ids,
                                latent_plan=plan_history,
                                film_mask=film_mask
                            )
                    else:
                        with torch.no_grad():
                            logits = model(generated_ids)

                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated_ids = torch.cat((generated_ids, next_token), dim=1)
                    last_token = generated_ids[0, -1].item()
                    new_action_tokens.append(next_token.item())
                    if last_token in [comma_id, eos_id]:
                        act_str = tokenizer.decode(new_action_tokens).replace(",", "")
                        act_str = act_str.replace("[DONE]", "").strip()
                        new_action_tokens = []
                        if act_str:
                            actions_taken.append(act_str)
                            if not sim.apply_action(act_str):
                                # print(f"Illegal move detected: {act_str}.")
                                # print(f"Generated trajectory: {', '.join(actions_taken)}")
                                # print(f"Final simulator state: {sim.state}")
                                return False, actions_taken

                    num_injected = 0
                    if last_token == comma_id and mistakes_injected < k_mistakes:
                        valid_moves = get_valid_actions(sim.state)
                        if valid_moves:
                            # mistake_tuple = random.choice(valid_moves)
                            mistake_tuple = valid_moves[-1]
                            mistake_action = mistake_tuple[0]
                            inject_str = mistake_action + ","
                            inject_ids = tokenizer.encode(inject_str,return_tensors="pt").to(device)
                            num_injected = inject_ids.size(1)
                            generated_ids = torch.cat((generated_ids, inject_ids), dim=1)
                            actions_taken.append(f"*{mistake_action}*")
                            sim.apply_action(mistake_action)
                            mistakes_injected += 1
                            # print(f"{tokenizer.decode(input_ids[0])}.. \
                            # with INJECTED {mistake_action} as forced mistake!")
                    last_token = generated_ids[0, -1].item()
                    if is_dynamic:
                        if last_token == comma_id:
                            encoder_outputs = model.planner.encoder(generated_ids,
                                                                    return_hidden_states=True)
                            act_emb = encoder_outputs[:, -1, :]
                            p_curr = model.planner.step_plan(act_emb, p_curr)

                        plans_to_add = p_curr.expand(-1, 1 + num_injected, -1)
                        plan_history = torch.cat((plan_history, plans_to_add), dim=1)
                    if last_token == eos_id:
                        break
                # print(f"Generated trajectory: {', '.join(actions_taken)}")
                # print(f"Final simulator state: {sim.state}")
                # breakpoint()
                return goal_conditions.issubset(sim.state), actions_taken

            dyn_success, dyn_traj = generate_trajectory(dynamic_model, is_dynamic=True,
                                                        init_str=init_str, prompt=prompt,
                                                        goal_conditions=goal_conditions,
                                                        k_mistakes=k_mistakes)
            base_success, base_traj = generate_trajectory(baseline, is_dynamic=False,
                                                          init_str=init_str, prompt=prompt,
                                                          goal_conditions=goal_conditions,
                                                          k_mistakes=k_mistakes)
            generated_trajectories.append({'prompt': prompt, 'dynamic': dyn_traj,
                                           'baseline': base_traj,
                                           'score': {'dynamic_success': dyn_success,
                                                     'baseline_success': base_success}})

            if dyn_success:
                dynamic_success += 1
            if base_success:
                baseline_success += 1

            # if idx == 0:
            #     print(f"  Dynamic Trajectory: {', '.join(dyn_traj)}")
            #     print(f"  Baseline Trajectory: {', '.join(base_traj)}")

        results["Dynamic"][k_mistakes] = (dynamic_success / num_samples) * 100
        results["Baseline"][k_mistakes] = (baseline_success / num_samples) * 100
        print(f"  Dynamic Model Success: {results['Dynamic'][k_mistakes]:.1f}%")
        print(f"  Baseline Model Success: {results['Baseline'][k_mistakes]:.1f}%")

        with open(f"./bw_benchmarks/resilience_results_{k_mistakes}.json",
                  "w", encoding="utf-8") as f:
            json.dump(generated_trajectories, f, indent=4)

    print("\n" + "="*50)
    print("FINAL RESILIENCE RESULTS")
    print(f"{'Perturbations':<15} | {'Baseline (AR)':<20} | {'Dynamic (Ours)':<20}")
    print("-" * 55)
    for k in perturbation_levels:
        print(f"{k:<15} | {results['Baseline'][k]:>18.1f}% | {results['Dynamic'][k]:>18.1f}%")

if __name__ == "__main__":
    run_k_perturbation_benchmark()
