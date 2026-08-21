""" Blocksworld Latent Control Evaluation Script """

import gc
import random
import json

import numpy as np
import torch
from transformers import set_seed

from .blocksworld_simulator import BWSimulator
from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.generate_bw_sub import generate_random_state, bfs_solve
from ..dataset.generate_bw_sub import BLOCKS

def generate_conflicting_goals():
    """Generates an init state and two goals that require different paths."""
    while True:
        init_state = generate_random_state(num_shuffles=20, active_blocks=BLOCKS)

        goal_a_state = generate_random_state(num_shuffles=5, active_blocks=BLOCKS)
        goal_b_state = generate_random_state(num_shuffles=5, active_blocks=BLOCKS)

        if init_state == goal_a_state or init_state == goal_b_state or goal_a_state == goal_b_state:
            continue

        ga_relaxed = set(cond for cond in goal_a_state if "on" in cond)
        gb_relaxed = set(cond for cond in goal_b_state if "on" in cond)

        path_a = bfs_solve(init_state, ga_relaxed, active_blocks=BLOCKS)
        path_b = bfs_solve(init_state, gb_relaxed, active_blocks=BLOCKS)

        if path_a and path_b and path_a[0] != path_b[0]:
            init_str = ", ".join(sorted(list(init_state)))
            goal_a_str = ", ".join(sorted(list(ga_relaxed)))
            goal_b_str = ", ".join(sorted(list(gb_relaxed)))
            return init_str, goal_a_str, goal_b_str

def run_controllability_benchmark():
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """Runs the latent controllability benchmark on the Blocksworld dataset."""
    num_samples=100

    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eval suite on {device}")
    set_seed(42)
    print(f"Loading Models on {device}...")

    ds_name = "blocksworld"
    dataset_config = get_dataset_config(name=ds_name)
    model_config = {
        'max_seq_length': 1024,
        'load_scratch': False,
        'dataset_name': ds_name,
        'custom_ar': True,
        'film': True,
        'vocab_size': dataset_config.get("vocab_size", 100),
        'tokenizer_path': dataset_config.get("tokenizer_path", ""),
    }
    base_working_model = 'gpt2_1024'
    model_config['load_stage'] = 'writer'
    model_config['working_model'] = base_working_model
    writer_model_path = f'{ds_name}_writer_model_{base_working_model}_goal_drop/writer_model.pt'
    model_config['writer_model_path'] = writer_model_path
    tokenizer, model = get_model_and_tokenizer(**model_config)
    model.to(device)
    model.eval()

    model_config['working_model'] = 'gpt2_1024-s'
    model_config['load_stage'] = 'base'
    _, base_model = get_model_and_tokenizer(**model_config)
    base_model.to(device)
    base_model.eval()

    num_samples=100
    comma_id = tokenizer.convert_tokens_to_ids(",")
    eos_id = tokenizer.convert_tokens_to_ids("[DONE]")

    conditions = ["Control_A", "Control_B", "Adversarial_Swap", "Blank_Goal"]

    for seed in [42, 1337, 2024, 7777, 9999]:
        results = {c: {"goal_A": 0, "goal_B": 0, "legal": 0} for c in conditions}
        set_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"\n=== RUNNING LATENT CONTROLLABILITY BENCHMARK ({num_samples} Samples {seed=}) ===")

        control_a_prompts = []
        control_b_prompts = []
        for i in range(num_samples):
            init_str, goal_a, goal_b = generate_conflicting_goals()
            cond_a = {c.strip() for c in goal_a.split(",")}
            cond_b = {c.strip() for c in goal_b.split(",")}

            for condition in conditions:
                sim = BWSimulator(init_str)
                decoder_prompt = ""
                if condition == "Control_A":
                    planner_prompt = f"[INIT] {init_str} [GOAL] {goal_a} "
                    decoder_prompt = f"[INIT] {init_str} [GOAL] {goal_a} "
                    control_a_prompts.append(planner_prompt)
                elif condition == "Control_B":
                    planner_prompt = f"[INIT] {init_str} [GOAL] {goal_b} "
                    decoder_prompt = f"[INIT] {init_str} [GOAL] {goal_b} "
                    control_b_prompts.append(planner_prompt)
                elif condition == "Adversarial_Swap":
                    planner_prompt = f"[INIT] {init_str} [GOAL] {goal_b} "
                    decoder_prompt = f"[INIT] {init_str} [GOAL] {goal_a} "
                elif condition == "Blank_Goal":
                    planner_prompt = f"[INIT] {init_str} [GOAL] {goal_b} "
                    decoder_prompt = f"[INIT] {init_str} [GOAL] "

                planner_ids = tokenizer.encode(planner_prompt, return_tensors="pt").to(device)
                decoder_ids = tokenizer.encode(decoder_prompt, return_tensors="pt").to(device)

                if decoder_ids.size(1) < planner_ids.size(1):
                    num_pads = planner_ids.size(1) - decoder_ids.size(1)
                    pad_tensor = torch.full((1, num_pads), tokenizer.pad_token_id,
                                            dtype=torch.long, device=device)
                    decoder_ids = torch.cat((decoder_ids, pad_tensor), dim=1)
                with torch.no_grad():
                    prompt_mask = torch.ones_like(planner_ids).to(device)
                    p_curr = model.planner.get_initial_plan(planner_ids, prompt_mask)

                is_legal = True
                generated_ids = model.generate(input_ids=decoder_ids, latent_plan=p_curr,
                                    comma_id=comma_id, eos_token_id=eos_id, max_new_tokens=50)

                prompt_len = decoder_ids.size(1)
                gen_text = tokenizer.decode(generated_ids[0][prompt_len:])
                gen_text = gen_text.replace("[DONE]", "").strip()
                actions = [a.strip() for a in gen_text.split(",") if a.strip()]
                for action in actions:
                    latest_action = action.strip()
                    if not sim.apply_action(latest_action):
                        is_legal = False
                        break

                if is_legal:
                    results[condition]["legal"] += 1
                if cond_a.issubset(sim.state) and is_legal:
                    results[condition]["goal_A"] += 1
                if cond_b.issubset(sim.state) and is_legal:
                    results[condition]["goal_B"] += 1

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_samples} trials...")

        print("\n" + "="*80)
        print("FINAL RESULTS: LATENT CONTROLLABILITY")
        print(f"{'Condition':<20} | {'Goal A Success':<15} |\
            {'Goal B Success':<15} | {'Legal Actions':<15}")
        print("-" * 80)
        for c in conditions:
            ga = (results[c]['goal_A'] / num_samples) * 100
            gb = (results[c]['goal_B'] / num_samples) * 100
            leg = (results[c]['legal'] / num_samples) * 100
            print(f"{c:<20} | {ga:>14.1f}% | {gb:>14.1f}% | {leg:>14.1f}%")

        # Baseline: Evaluate the base model on the same prompts for comparison
        # with open(f"eval_prompts_latent_control_{seed}.json", "r") as f:
        #     prompt_data = json.load(f)
        # control_a_prompts = prompt_data["Control_A"]
        # control_b_prompts = prompt_data["Control_B"]
        base_results = {c: {"goal_A": 0, "goal_B": 0, "legal": 0}
                        for c in ["Control_A", "Control_B"]
                        }
        print("\n=== EVALUATING BASELINE MODEL ON SAME PROMPTS ===")
        for condition, prompts in [("Control_A", control_a_prompts),
                                   ("Control_B", control_b_prompts)]:
            legal_count = 0
            success_a_count = 0
            success_b_count = 0
            for prompt in prompts:
                init_str = prompt.split("[GOAL]")[0].replace("[INIT]", "").strip()
                goal_str = prompt.split("[GOAL]")[1].strip()
                goal_conditions = {c.strip() for c in goal_str.split(",") if c.strip()}

                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_ids = base_model.generate(input_ids=input_ids, max_new_tokens=50)
                prompt_len = input_ids.size(1)
                raw_text = tokenizer.decode(generated_ids[0][prompt_len:])
                raw_text = raw_text.split("[DONE]")[0].strip()
                gen_text = raw_text.replace("[DONE]", "").strip()
                actions = [a.strip() for a in gen_text.split(",") if a.strip()]
                sim = BWSimulator(init_str)
                is_legal = True
                for action in actions:
                    if not sim.apply_action(action):
                        is_legal = False
                        break

                if is_legal:
                    legal_count += 1
                if condition == "Control_A" and goal_conditions.issubset(sim.state) and is_legal:
                    success_a_count += 1
                if condition == "Control_B" and goal_conditions.issubset(sim.state) and is_legal:
                    success_b_count += 1

            base_results[condition] = {"goal_A": success_a_count,
                                       "goal_B": success_b_count,
                                       "legal": legal_count
                                       }
            print(f"\nBaseline Results for {condition}:")
            print(f"Legal Action Trajectories: {(legal_count / len(prompts)) * 100:.1f}%")
            print(f"Goal A Success Rate: {(success_a_count / len(prompts)) * 100:.1f}%")
            print(f"Goal B Success Rate: {(success_b_count / len(prompts)) * 100:.1f}%")

        final_results = {"Ours": results, "Baseline": base_results}
        with open(f"./bw_benchmarks/latent_control_results_{seed}.json", "w",
                  encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        with open(f"./bw_benchmarks/latent_control_prompts_{seed}.json", "w",
                  encoding="utf-8") as f:
            json.dump({"Control_A": control_a_prompts, "Control_B": control_b_prompts}, f, indent=4)

def mean_results():
    """Computes the mean and std of results across seeds."""
    conditions = ["Control_A", "Control_B", "Adversarial_Swap", "Blank_Goal"]
    results = {c: {"goal_A": [], "goal_B": [], "legal": []} for c in conditions}
    for seed in [42, 1337, 2024, 7777, 9999]:
        with open(f"./bw_benchmarks/latent_control_results_{seed}.json", "r",
                  encoding="utf-8") as f:
            data = json.load(f)
        for c in data["Ours"].keys():
            results[c]['goal_A'].append(data["Ours"][c]['goal_A'])
            results[c]['goal_B'].append(data["Ours"][c]['goal_B'])
            results[c]['legal'].append(data["Ours"][c]['legal'])

    print("\n=== MEAN RESULTS FOR Ours ===")
    for c in conditions:
        mean_ga = np.mean(results[c]['goal_A'])
        std_ga = np.std(results[c]['goal_A'])
        mean_gb = np.mean(results[c]['goal_B'])
        std_gb = np.std(results[c]['goal_B'])
        mean_leg = np.mean(results[c]['legal'])
        std_leg = np.std(results[c]['legal'])
        print(f"{c:<20} | {mean_ga:>14.1f}% ± {std_ga:.1f} |\
              {mean_gb:>14.1f}% ± {std_gb:.1f} | {mean_leg:>14.1f}% ± {std_leg:.1f}")

    print("" + "-"*80)
    conditions = ["Control_A", "Control_B"]
    base_results = {c: {"goal_A": [], "goal_B": [], "legal": []} for c in conditions}
    for seed in [42, 1337, 2024, 7777, 9999]:
        with open(f"./bw_benchmarks/latent_control_results_{seed}.json", "r",
                  encoding="utf-8") as f:
            data = json.load(f)
        for c in data["Baseline"].keys():
            base_results[c]['goal_A'].append(data["Baseline"][c]['goal_A'])
            base_results[c]['goal_B'].append(data["Baseline"][c]['goal_B'])
            base_results[c]['legal'].append(data["Baseline"][c]['legal'])

    print("\n=== MEAN RESULTS FOR BASELINE ===")
    for c in conditions:
        mean_ga = np.mean(base_results[c]['goal_A'])
        std_ga = np.std(base_results[c]['goal_A'])
        mean_gb = np.mean(base_results[c]['goal_B'])
        std_gb = np.std(base_results[c]['goal_B'])
        mean_leg = np.mean(base_results[c]['legal'])
        std_leg = np.std(base_results[c]['legal'])
        print(f"{c:<20} | {mean_ga:>14.1f}% ± {std_ga:.1f} |\
              {mean_gb:>14.1f}% ± {std_gb:.1f} | {mean_leg:>14.1f}% ± {std_leg:.1f}")

if __name__ == "__main__":
    run_controllability_benchmark()
    mean_results()
