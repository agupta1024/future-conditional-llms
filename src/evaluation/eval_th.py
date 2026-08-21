"""Treasure Hunt Evaluation Suite"""
import gc
import os
import re
import json
import random
from collections import defaultdict

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import set_seed

from .plot_metrics import treasure_hunt_metric_analysis, profile_model_efficiency
from .plot_metrics import calculate_efficiency
from ..config.model_config import get_model_and_tokenizer
from ..config.dataset_config import get_dataset_config
from ..dataset.prepare_dataset_th import get_dataloaders as get_th_dataloaders
from ..dataset.generate_th import generate_filler, ITEMS, ROOMS

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def print_model_info(model, name="Model"):
    """Prints the number of parameters and architecture dimensions of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- {name} ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    if hasattr(model, 'planner'):
        planner_params = sum(p.numel() for p in model.planner.parameters())
        print(f"Planner Parameters:   {planner_params:,}")

    if hasattr(model, 'config'):
        print(f"Layers (n_layer):     {getattr(model.config, 'n_layer', 'N/A')}")
        print(f"Hidden Dim (n_embd):  {getattr(model.config, 'n_embd', 'N/A')}")
        print(f"Heads (n_head):       {getattr(model.config, 'n_head', 'N/A')}")

def generate_horizon_tsne(baseline_model, ours_model, tokenizer, device, horizon=1024):
    # pylint: disable=too-many-locals, too-many-statements
    """
    Generates a side-by-side t-SNE plot of the decoder's hidden states 
    at extreme horizons to visualize amnesia vs. global retention.
    """
    print(f"\n--- GENERATING DEEP HORIZON t-SNE (H={horizon}) ---")
    baseline_model.eval()
    ours_model.eval()

    categories = ['Ruby Amulet', 'Iron Key', 'Ancient Scroll']
    colors = ['#cc6633', '#285c9e', '#2ca02c']

    all_texts = []
    labels = []
    horizon_to_target_length = {128: 64, 256: 156, 512: 370, 1024: 830}
    target_length = horizon_to_target_length.get(horizon, 200)
    print("Generating long-horizon trajectories...")
    for cat in categories:
        for _ in range(100):
            code = np.random.randint(1000, 9999)
            prompt = f"[GOAL] Target: {cat} | Code: {code}\n[TRAJECTORY] "

            filler_sentences = generate_filler(target_length)
            filler = " ".join(filler_sentences) + " "

            trigger = "I checked my map and realized the vault was hidden in the Library."
            action_prefix = " I finally arrived at the Library. I approached the lockpad "
            action_prefix += f"and entered {code}. The vault clicked open, and I retrieved the "

            full_text = prompt + filler + trigger + action_prefix

            all_texts.append(full_text)
            labels.append(cat)

    tokenizer.padding_side = 'left'
    encoded = tokenizer(all_texts, return_tensors="pt", padding=True).to(device)

    max_positions = 1024
    if encoded.input_ids.size(1) > max_positions:
        input_ids = encoded.input_ids[:, -max_positions:]
    else:
        input_ids = encoded.input_ids

    input_ids = torch.clamp(input_ids, min=0, max=50256)
    print(f"Extracting vectors at the horizon {horizon} (Sequence Length: {input_ids.size(1)})...")

    context_w = 500
    baseline_input_ids = input_ids[:, -context_w:]
    print(f"Extracting hidden states at the horizon {horizon}...")
    with torch.no_grad():
        baseline_outputs = baseline_model(baseline_input_ids, return_hidden_states=True)
        last_layer_hidden_states = baseline_outputs
        baseline_vector = last_layer_hidden_states[:, -1, :].cpu().numpy()

        prompt_texts = [text.split("[TRAJECTORY]")[0] + "[TRAJECTORY] " for text in all_texts]
        tokenizer.padding_side = 'right'
        prompt_encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
        latent_plan, _, _ = ours_model.planner(prompt_encoded.input_ids,
                                               prompt_encoded.attention_mask)
        if len(latent_plan.shape) == 2:
            latent_plan = latent_plan.unsqueeze(1)

        ours_decoder_out = ours_model(baseline_input_ids, latent_plan=latent_plan,
                                       return_hidden_states=True)
        ours_vector = ours_decoder_out[:, -1, :].cpu().numpy()

    print("Running t-SNE mapping...")
    tsne_base = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    vectors_base = tsne_base.fit_transform(baseline_vector)

    tsne_ours = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    vectors_ours = tsne_ours.fit_transform(ours_vector)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    for i, cat in enumerate(categories):
        ax1.scatter(vectors_base[i*100:(i+1)*100, 0], vectors_base[i*100:(i+1)*100, 1],
                    c=colors[i], label=cat, alpha=0.8, edgecolors='w', s=60)

        ax2.scatter(vectors_ours[i*100:(i+1)*100, 0], vectors_ours[i*100:(i+1)*100, 1],
                    c=colors[i], label=cat, alpha=0.8, edgecolors='w', s=60)

    ax1.set_title(f"Baseline GPT-2 at H={horizon}", fontsize=14, pad=15)
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax1.legend(frameon=True, shadow=True)

    ax2.set_title(f"Ours (FiLM) at H={horizon}", fontsize=14, pad=15)
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    ax2.legend(frameon=True, shadow=True)

    filename = f"./th_benchmarks/deep_horizon_tsne_H{horizon}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Success! Saved deep horizon t-SNE plot to '{filename}'")

def evaluate_model_efficiency(model_name, model, device, val_dataloader,
                              context_windows):
    # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-branches
    """
    Evaluates the efficiency of the model in terms of inference time and memory usage.
    """
    model.eval()
    c_bin = {f"context_{c}": defaultdict(list) for c in context_windows}
    memory_usage = {
        "tokens_per_second": [],
        "peak_memory_mb": [],
        "total_time_sec": [],
    }
    for context_window in context_windows:
        print(f"Evaluating efficiency for context window: {context_window}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= 10:
                    break
                input_ids = batch['prompt_ids'].to(device)
                metrics = profile_model_efficiency(model_name, model,
                                                   input_ids, max_new_tokens=512,
                                                   context_window=context_window)
                for key, value in metrics.items():
                    memory_usage[key].append(value)

        c_bin[f"context_{context_window}"] = {k: sum(v) / len(v) for k, v in memory_usage.items()}
    return c_bin

def evaluate_horizon_memory(model_name, model, tokenizer, device,
                            horizons, context_windows, num_samples=100):
    # pylint: disable=too-many-locals, too-many-positional-arguments, too-many-arguments
    """
    Evaluates Semantic Retention, Exact Match.
    """
    model.eval()
    horizon_to_input_ids_length = {64: 16, 128: 64, 256: 156, 512: 370, 1024: 830}
    c_bin = {f"context_{c}": defaultdict(list) for c in context_windows}
    for context_window in context_windows:
        h_bin = {f"horizon_{h}": defaultdict(list) for h in horizons}
        for h in horizons:
            target_length = horizon_to_input_ids_length[h]
            metrics = {
                "target_retrieval": 0,
                "passkey_retrieval": 0,
                "exact_match": 0,
            }

            for _ in range(num_samples):
                item = random.choice(ITEMS)
                room = random.choice(ROOMS)
                code = f"{random.randint(1000, 9999)}"
                prompt = f"[GOAL] Target: {item} | Code: {code}\n[TRAJECTORY] "
                filler_sentences = generate_filler(target_length)
                clue = f"I checked my map and realized the vault was hidden in the {room}."
                filler_sentences.append(clue)
                full_context = prompt + " ".join(filler_sentences)
                input_ids = tokenizer.encode(full_context, return_tensors="pt").to(device)
                with torch.no_grad():
                    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    prompt_mask = torch.ones_like(prompt_ids).to(device)

                    if model_name == "Baseline":
                        latent_plan = None
                    else:
                        latent_plan, _, _ = model.planner(prompt_ids, prompt_mask)
                        if len(latent_plan.shape) == 2:
                            latent_plan = latent_plan.unsqueeze(1)

                    generated_ids = model.generate(input_ids,
                                                   latent_plan=latent_plan,
                                                   tokenizer=tokenizer,
                                                   max_new_tokens=100,
                                                   context_window=context_window,
                                                   eos_token_id=tokenizer.eos_token_id,
                                                   pad_token_id=tokenizer.pad_token_id)
                input_length = input_ids.shape[1]
                generated_text = tokenizer.decode(generated_ids[0][input_length:])
                # print(f"Context Window: {context_window} | Horizon: {h}")
                # print(f"Input Length: {input_length} |\
                # Generated Length: {generated_ids.shape[1] - input_length}")
                # print(f"Prompt: {prompt}")
                # print(f"Generated Text:\n{generated_text}\n{'-'*50}")
                # breakpoint()
                item_found = item.lower() in generated_text.lower()
                if item_found:
                    metrics["target_retrieval"] += 1

                code_match = re.search(r'\b\d{4}\b', generated_text)
                gen_code = code_match.group(0) if code_match else None
                code_exact = gen_code == code

                if code_exact:
                    metrics["passkey_retrieval"] += 1

                if item_found and code_exact:
                    metrics["exact_match"] += 1

            h_bin[f"horizon_{h}"] = {k: (v / num_samples) * 100 for k, v in metrics.items()}
        c_bin[f"context_{context_window}"] = h_bin
    return c_bin

def main(): # pylint: disable=too-many-locals, too-many-statements
    """Main function to run the Treasure Hunt evaluation suite."""
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treasure Hunt Eval suite on {device}")
    set_seed(42)

    ds_name = "treasure_hunt"

    dataset_config = get_dataset_config(name = ds_name)
    validation_path = dataset_config.get("val_path", "")
    print(f"Loading {ds_name} dataset from {validation_path}...")

    _, val_dataloader,_,_ = get_th_dataloaders(
        train_path=None,
        eval_path=validation_path,
        batch_size=32
    )

    model_config = {
        'max_seq_length': 512,
        'load_scratch': False,
        'dataset_name': ds_name,
        'custom_ar': True,
        'film': True,
        'vocab_size': dataset_config.get("vocab_size", 50257),
        'tokenizer_path': dataset_config.get("tokenizer_path", ""),
    }

    model_config['load_stage'] = "writer"
    model_config['working_model'] = 'gpt2_512'
    tokenizer, ours_model = get_model_and_tokenizer(**model_config)
    ours_model.to(device)
    ours_model.eval()
    print_model_info(ours_model, "Ours Model")

    model_config['load_stage'] = "base"
    model_config['working_model'] = 'gpt2_512-l'
    tokenizer, baseline_model = get_model_and_tokenizer(**model_config)
    baseline_model.to(device)
    baseline_model.eval()
    print_model_info(baseline_model, "Baseline Model")

    horizons = [128, 256, 512, 1024]
    context_windows = [128, 256, 500]

    model_names = ['Ours', 'Baseline']
    aggregated_data = {
        "context_windows": context_windows,
        "horizons": horizons,
        "models": {model_name: defaultdict(lambda: defaultdict(list)) for model_name in model_names}
    }
    efficiency_metrics = {
        "context_windows": context_windows,
        "horizons": horizons,
        "models": {model_name: defaultdict(lambda: defaultdict(list)) for model_name in model_names}
    }
    default_seeds = [42, 1337, 2024, 7777, 9999]

    for seed in default_seeds:
        set_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for model_name in model_names:
            if model_name == "Ours":
                model = ours_model
            else:
                model = baseline_model
            print(f"\n[!] Benchmarking '{model_name}' model with seed {seed}...")

            efficiency_metrics["models"][model_name] = evaluate_model_efficiency(model_name, model,
                                                                device, val_dataloader,
                                                                context_windows=context_windows)
            aggregated_data["models"][model_name] = evaluate_horizon_memory(model_name, model,
                                                        tokenizer, device, horizons=horizons,
                                                        context_windows=context_windows,
                                                        num_samples=100)

        final_output = {
            "horizons": horizons,
            "context_windows": context_windows,
            "models": {model_name: defaultdict(lambda: defaultdict(list))
                       for model_name in model_names}
        }
        for context_window in context_windows:
            for model_name in model_names:
                final_output["models"][model_name][f"context_{context_window}"] = defaultdict(list)

        raw_retention_data = {}
        for model_name, c_bins in aggregated_data["models"].items():
            raw_retention_data[model_name] = {}
            for c_str, h_bins in c_bins.items():
                raw_retention_data[model_name][c_str] = {}
                for h_str, lists in h_bins.items():
                    final_output["models"][model_name][c_str][h_str] = {
                        "target_retrieval": lists.get("target_retrieval", 0),
                        "passkey_retrieval": lists.get("passkey_retrieval", 0),
                        "exact_match": lists.get("exact_match", 0)
                    }
        for model_name, c_bins in efficiency_metrics["models"].items():
            for c_str, metrics in c_bins.items():
                final_output["models"][model_name][c_str]["efficiency"] = {
                    "tokens_per_second": metrics.get("tokens_per_second", 0),
                    "peak_memory_mb": metrics.get("peak_memory_mb", 0),
                    "total_time_sec": metrics.get("total_time_sec", 0)
                }
        os.makedirs("./th_benchmarks", exist_ok=True)
        with open(f"./th_benchmarks/eval_results_{seed}.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)

    generate_horizon_tsne(baseline_model=baseline_model, ours_model=ours_model,
                          tokenizer=tokenizer, device=device, horizon=1024)
    treasure_hunt_metric_analysis(filetag="./th_benchmarks/eval_results", seeds=default_seeds)
    calculate_efficiency(filetag="./th_benchmarks/eval_results", seeds=default_seeds)


if __name__ == "__main__":
    main()
