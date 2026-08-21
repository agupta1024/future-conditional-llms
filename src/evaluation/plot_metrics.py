""" Plotting Functions for Treasure Hunt Evaluation """

import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300
})
# =====================================================================
# Computational Efficiency (Memory & Speed)
# =====================================================================
def profile_model_efficiency(model_name, model, input_ids,
                             max_new_tokens=512, comma_id=None,
                             context_window=None):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Measures the empirical inference speed (tokens/sec) and peak VRAM usage.
    """
    model.eval()
    device = input_ids.device

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        if model_name == "Ours":
            # print("Profiling with future ids provided...")
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
                eos_token_id=-1,
                comma_id=comma_id,
            )
        else:
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
                eos_token_id=-1,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    generation_time = end_time - start_time

    batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
    total_generated_tokens = batch_size * max_new_tokens

    tokens_per_second = total_generated_tokens / generation_time

    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)

    return {
        "tokens_per_second": tokens_per_second,
        "peak_memory_mb": peak_memory_mb,
        "total_time_sec": generation_time
    }

# =====================================================================
# Statistical Significance (Confidence Intervals)
# =====================================================================
def calculate_bounds_for_metric(metric_data, horizons):
    """ Calculates the mean and 95% Confidence Interval for a metric across multiple runs/seeds."""
    means = []
    lower_bounds = []
    upper_bounds = []
    std_dev = []

    for h in horizons:
        m, lb, ub = calculate_confidence_intervals(metric_data[h])
        std_dev.append(np.std(np.array(metric_data[h]), ddof=1).item())  # Sample standard deviation
        means.append(m)
        lower_bounds.append(lb)
        upper_bounds.append(ub)
    return lower_bounds, means, upper_bounds, std_dev

def calculate_confidence_intervals(metric_data, confidence=0.95):
    """
    Calculates the mean and 95% Confidence Interval for a metric across multiple runs/seeds.
    
    Args:
        metric_data: A 1D array or list of metric scores (e.g., retention percentages 
                     from 5 different random seeds, or 100 validation batches).
                     
    Returns:
        mean, lower_bound, upper_bound
    """
    data = np.array(metric_data)
    n = len(data)
    mean = np.mean(data).item()
    if n < 2:
        return mean, mean, mean
    sem = st.sem(data) # Standard Error of the Mean
    if sem == 0:
        return mean, mean, mean

    margin_of_error = sem * st.t.ppf((1 + confidence) / 2., n-1)

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return mean, lower_bound, upper_bound

def pretty_print_metrics(data_ours, data_baseline, target_context=128,
                        metric_name="target_retrieval", horizon=None):
    """
    Print metrics in a structured format.
    """
    horizons = sorted(list(data_ours[metric_name][target_context].keys()))
    _, mean_ours, _, std_ours = calculate_bounds_for_metric(
        data_ours[metric_name][target_context], horizons=horizons)
    _, mean_base, _, std_base = calculate_bounds_for_metric(
        data_baseline[metric_name][target_context], horizons=horizons)

    pc_delta = np.divide(
        (np.array(mean_ours) - np.array(mean_base)) * 100,
        np.array(mean_base),
        out=np.zeros_like(np.array(mean_base), dtype=float),
        where=np.array(mean_base) != 0
    )
    pc_delta = np.round(pc_delta, 2)
    std_ours = np.round(np.array(std_ours), 2)
    std_base = np.round(np.array(std_base), 2)

    print("" + "="*80)
    print(f"Treasure Hunt: {metric_name} at Context W={target_context}")
    for h, m_ours, std_ours, m_base, std_base, pc in zip(horizons, mean_ours, std_ours,
                                                         mean_base, std_base, pc_delta):
        if horizon is not None and h not in horizon:
            continue
        print("Horizon | Ours         | Baseline    | % Change")
        print(f"{h}    | {m_ours:.2f} + {std_ours:.2f} | {m_base:.2f} + {std_base:.2f} | {pc:.2f}%")
        print("" + "-"*80)
    print("" + "="*80)

def generate_th_retention_plot(data_ours, data_baseline, target_context=128,
                               metric_name="target_retrieval", horizon=None):
    # pylint: disable=too-many-locals
    """
    Generates Semantic Retention vs Horizon for the Treasure Hunt benchmark.
    """
    print(f"Generating Treasure Hunt {metric_name} Plot...")
    horizons = sorted(list(data_ours[metric_name][target_context].keys()))

    lower_ours, mean_ours, upper_ours, _ = calculate_bounds_for_metric(
        data_ours[metric_name][target_context], horizons=horizons)
    lower_base, mean_base, upper_base, _ = calculate_bounds_for_metric(
        data_baseline[metric_name][target_context], horizons=horizons)
    pretty_print_metrics(data_ours, data_baseline, target_context=target_context,
                        metric_name=metric_name, horizon=horizon)

    plt.style.use('seaborn-v0_8-whitegrid')
    _, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(horizons, mean_base, marker='s', markersize=8, linewidth=2.5,
            color='#cc6633', label='Baseline (AR GPT-2)')
    ax.fill_between(horizons, lower_base, upper_base, color='#cc6633', alpha=0.2, edgecolor='none')

    ax.plot(horizons, mean_ours, marker='o', markersize=8, linewidth=2.5,
            color='#285c9e', label='Ours (Global Conditioning)')
    ax.fill_between(horizons, lower_ours, upper_ours, color='#285c9e', alpha=0.2, edgecolor='none')

    ax.set_xscale('log', base=2)
    ax.set_xticks(horizons)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_ylim(0, 105)
    ax.set_ylabel('Semantic Retention (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation Horizon ($H$ Tokens)', fontsize=12, fontweight='bold')
    ax.set_title(f'Treasure Hunt: Semantic Retention (Context $W={target_context}$)',
                 fontsize=14, pad=15)

    ax.axvline(x=target_context, color='grey', linestyle='--', alpha=0.7,
               label=f'Context Boundary ($W={target_context}$)')

    ax.legend(loc='lower left', fontsize=8, frameon=True, shadow=True)

    plt.tight_layout()
    filename = f'./th_benchmarks/th_{metric_name}_W{target_context}.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def aggregate_th_results_across_seeds(filetag, metric_name, seeds, skip_horizon=None):
    # pylint: disable=too-many-locals, too-many-branches
    """
    Aggregates results from multiple seeds into a structured format for plotting.
    Expects files named like: benchmark_results_42.json, benchmark_results_1337.json, etc.
    """
    c_h_ours = {}
    h_c_ours = {}
    for seed in seeds:
        filename_ours = f"{filetag}_{seed}.json"
        with open(filename_ours, "r", encoding="utf-8") as f:
            data_ours = json.load(f)
            for c in data_ours["context_windows"]:
                c_h_ours[(c)] = c_h_ours.get((c), {})
                for h in data_ours["horizons"]:
                    if skip_horizon is None or h not in skip_horizon:
                        c_h_ours[(c)][(h)] = c_h_ours[(c)].get((h), [])
                        c_h_ours[(c)][(h)].append(data_ours["models"]["Ours"]
                                                  [f"context_{c}"][f"horizon_{h}"][metric_name])
            for h in data_ours["horizons"]:
                h_c_ours[(h)] = h_c_ours.get((h), {})
                for c in data_ours["context_windows"]:
                    if skip_horizon is None or h not in skip_horizon:
                        h_c_ours[(h)][(c)] = h_c_ours[(h)].get((c), [])
                        h_c_ours[(h)][(c)].append(data_ours["models"]["Ours"]
                                                  [f"context_{c}"][f"horizon_{h}"][metric_name])
    c_h_baseline = {}
    h_c_baseline = {}
    for seed in seeds:
        filename_baseline = f"{filetag}_{seed}.json"
        with open(filename_baseline, "r", encoding="utf-8") as f:
            data_baseline = json.load(f)
            for c in data_baseline["context_windows"]:
                c_h_baseline[(c)] = c_h_baseline.get((c), {})
                for h in data_baseline["horizons"]:
                    if skip_horizon is None or h not in skip_horizon:
                        c_h_baseline[(c)][(h)] = c_h_baseline[(c)].get((h), [])
                        c_h_baseline[(c)][(h)].append(data_baseline["models"]["Baseline"]
                                                      [f"context_{c}"][f"horizon_{h}"][metric_name])
            for h in data_baseline["horizons"]:
                h_c_baseline[(h)] = h_c_baseline.get((h), {})
                for c in data_baseline["context_windows"]:
                    if skip_horizon is None or h not in skip_horizon:
                        h_c_baseline[(h)][(c)] = h_c_baseline[(h)].get((c), [])
                        h_c_baseline[(h)][(c)].append(data_baseline["models"]["Baseline"]
                                                      [f"context_{c}"][f"horizon_{h}"][metric_name])

    return c_h_ours, c_h_baseline

def calculate_efficiency(filetag, seeds):
    # pylint: disable=too-many-branches, too-many-locals
    """Calculate efficiency metrics for both ours and baseline models."""
    c_eff_ours = {}
    c_eff_baseline = {}
    for seed in seeds:
        filename_ours = f"{filetag}_{seed}.json"
        with open(filename_ours, "r", encoding="utf-8") as f:
            data_ours = json.load(f)
            for c in data_ours["context_windows"]:
                c_eff_ours[(c)] = c_eff_ours.get((c), {})
                ours_dict = data_ours["models"]["Ours"][f"context_{c}"]
                efficiency_metrics = ours_dict["efficiency"]
                for metric_name, val in efficiency_metrics.items():
                    c_eff_ours[(c)][(metric_name)] = c_eff_ours[(c)].get((metric_name), [])
                    c_eff_ours[(c)][(metric_name)].append(val)

    for seed in seeds:
        filename_baseline = f"{filetag}_{seed}.json"
        with open(filename_baseline, "r", encoding="utf-8") as f:
            data_baseline = json.load(f)
            for c in data_baseline["context_windows"]:
                c_eff_baseline[(c)] = c_eff_baseline.get((c), {})
                baseline_dict = data_baseline["models"]["Baseline"][f"context_{c}"]
                efficiency_metrics = baseline_dict["efficiency"]
                for metric_name, val in efficiency_metrics.items():
                    c_eff_baseline[(c)][(metric_name)] = c_eff_baseline[(c)].get((metric_name), [])
                    c_eff_baseline[(c)][(metric_name)].append(val)

    for c in c_eff_ours:
        c_eff_ours[(c)] = {metric_name: [np.mean(values).item(), np.std(values, ddof=1).item()]
                           for metric_name, values in c_eff_ours[(c)].items()}
    for c in c_eff_baseline:
        c_eff_baseline[(c)] = {metric_name: [np.mean(values).item(), np.std(values, ddof=1).item()]
                               for metric_name, values in c_eff_baseline[(c)].items()}
    metric_names = list(c_eff_ours.keys())
    for c, data in c_eff_ours.items():
        print(f"Context Window: {c}")
        print("Ours Efficiency Metrics:")
        for metric_name, (mean, std) in data.items():
            print(f"  {metric_name}: Mean={mean:.4f}, Std={std:.4f}")
        print("Baseline Efficiency Metrics:")
        for metric_name, (mean, std) in c_eff_baseline[(c)].items():
            print(f"  {metric_name}: Mean={mean:.4f}, Std={std:.4f}")
        print("Percentage Change (Ours vs Baseline):")
        for metric_name in metric_names:
            mean_ours, _ = data[(metric_name)]
            mean_base, _ = c_eff_baseline[(c)][(metric_name)]
            if mean_base != 0:
                pc_delta = (mean_ours - mean_base) * 100 / mean_base
                print(f"  {metric_name}: Percentage Change: {pc_delta:.2f}%")
            else:
                print(f"  {metric_name}: Baseline mean is zero, cannot compute percentage change.")

    return c_eff_ours, c_eff_baseline

def treasure_hunt_metric_analysis(filetag, seeds,
                                  skip_horizon=None,
                                  horizon=None):
    """
    Aggregates and plots metrics for the Treasure Hunt benchmark.
    """
    metric_map = {
        "target_retrieval": ["Target Retrieval (%)", "Retention", 105],
        "passkey_retrieval": ["Passkey Retrieval (%)", "Retention", 105],
        "exact_match": ["Exact Match (%)", "Repetition Rate", 0.1]
    }
    agg_metrics = {"Ours": {}, "Baseline": {}}

    for metric in metric_map:
        agg_metrics["Ours"][metric] = {}
        agg_metrics["Baseline"][metric] = {}
        ours, baseline = aggregate_th_results_across_seeds(
                                        filetag=filetag,
                                        metric_name=metric,
                                        seeds=seeds,
                                        skip_horizon=skip_horizon)
        agg_metrics["Ours"][metric] = ours
        agg_metrics["Baseline"][metric] = baseline
        generate_th_retention_plot(agg_metrics["Ours"], agg_metrics["Baseline"],
                                   target_context=500, metric_name=metric, horizon=horizon)

if __name__ == "__main__":
    treasure_hunt_metric_analysis(filetag="./th_benchmarks/eval_results",
                                  seeds=[42, 1337, 2024, 7777, 9999], horizon=[128, 1024])
    # calculate_efficiency(filetag="./th_benchmarks/eval_results",
    #                      seeds=[42, 1337, 2024, 7777, 9999])
