""" Blocksworld Model Evaluation Script"""

import os
import re
import json
import torch
from transformers import set_seed

from .blocksworld_simulator import BWSimulator
from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer

def evaluate_dynamic_model(model, tokenizer, eval_data_path, is_dynamic=False,
                           num_samples=500, device=torch.device("cpu")):
    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches, too-many-positional-arguments
    """ Evaluate the model on the Blocksworld dataset. """
    comma_id = tokenizer.convert_tokens_to_ids(",")
    eos_id = tokenizer.convert_tokens_to_ids("[DONE]")

    print(f"\n=== EVALUATING MODEL ({num_samples} SAMPLES) ===")

    total_tested = 0
    eval_results = []

    total_success = 0
    total_partial = 0.0
    total_legal = 0
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            prompt = data["prompt"] + " "

            if BWSimulator is not None:
                init_str = prompt.split("[GOAL]")[0].replace("[INIT]", "").strip()
                goal_str = prompt.split("[GOAL]")[1].strip()

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                if is_dynamic:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        comma_id=comma_id,
                        eos_id=eos_id,
                        max_new_tokens=50,
                    )
                else:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=50,
                        eos_token_id=eos_id,
                    )
            output_text = tokenizer.decode(generated_ids[0])
            gen_text = re.sub(r" ,", ",", output_text)
            gen_text = gen_text[len(prompt):]
            for token in ["[DONE]", "[PAD]", "<|endoftext|>"]:
                if token in gen_text:
                    gen_text = gen_text.split(token)[0]

            actions = [act.strip() for act in gen_text.split(",") if act.strip()]
            total_tested += 1

            # if total_tested <= 3:
            #     print("\n" + "="*50)
            #     print(f"PROMPT: {prompt}")
            #     print(f"DYNAMIC PLAN GENERATED: {', '.join(actions)}")
            #     print(f"Raw Output: {output_text}")

            if BWSimulator is not None:
                sim = BWSimulator(init_str)
                # print(f"\n[Actions]: {actions}")
                for action in actions:
                    # print(f"Applying action: {action.strip()}")
                    if not sim.apply_action(action):
                        break

                score = sim.score_goal(goal_str)
                total_success += int(score["success"])
                total_partial += score["partial_pct"]
                total_legal += int(score["legal_execution"])
                eval_results.append((prompt, gen_text, score))

            if total_tested >= num_samples:
                break

    results = {
        "goal_success_rate": (total_success / num_samples) * 100 if num_samples > 0 else 0,
        "legal_action_rate": (total_legal / num_samples) * 100 if num_samples > 0 else 0,
        "average_partial_completion": (total_partial / num_samples) if num_samples > 0 else 0
    }
    print(f"Goal Success Rate (Perfect Plan): {(total_success / num_samples) * 100:.1f}%")
    print(f"Legal Action Trajectories:        {(total_legal / num_samples) * 100:.1f}%")
    print(f"Average Partial Completion:       {total_partial / num_samples:.1f}%")
    return results

def main():
    """ Main function to evaluate the dynamic model and baseline model on Blocksworld dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading environment on {device}...")
    set_seed(42)

    base_working_model = 'gpt2_1024'
    ds_name = "blocksworld_lexical"
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

    results = evaluate_dynamic_model(dynamic_model, tokenizer, eval_data_path,
                           is_dynamic=True, num_samples=500, device=device)

    base_working_model = 'gpt2_1024-l'
    model_config['working_model'] = base_working_model
    model_config['load_stage'] = 'base'
    _, baseline = get_model_and_tokenizer(**model_config)
    baseline.to(device)
    baseline.eval()
    baseline_results = evaluate_dynamic_model(baseline, tokenizer, eval_data_path,
                               is_dynamic=False, num_samples=500, device=device)
    final_results = {
        "dynamic_model": results,
        "baseline_model": baseline_results
    }
    os.makedirs("./bw_benchmarks", exist_ok=True)
    with open("./bw_benchmarks/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)


if __name__ == "__main__":
    main()
