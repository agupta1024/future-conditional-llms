""" Blocksworld Latent Control Evaluation Script """

import gc
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import set_seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .blocksworld_simulator import BWSimulator
from ..config.dataset_config import get_dataset_config
from ..config.model_config import get_model_and_tokenizer
from ..dataset.generate_bw_sub import generate_random_state, bfs_solve
from ..dataset.generate_bw_sub import BLOCKS
from sklearn.decomposition import PCA
from torch.nn import functional as F

def extract_layer_states(model, input_ids, latent_plan):
    layer_states = []
    handles = []

    def make_hook(idx):
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                layer_states.append(h[0, -1, :].detach())

        return hook

    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(input_ids=input_ids, latent_plan=latent_plan)

    for h in handles:
        h.remove()

    return torch.stack(layer_states)

def generate_conflicting_goals(init_state=None):
    """Generates an init state and two goals that require different paths."""
    while True:
        if init_state is None:
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
def extract_single_layer_state(model, input_ids, latent_plan, layer_idx):
    """Extracts the last-token hidden state of a specific layer for a single input pass.

    Args:
        model: The PyTorch transformer model.
        input_ids: Tensor of token IDs [1, seq_len].
        latent_plan: Continuous FiLM plan tensor.
        layer_idx: Int, the target layer index to extract (e.g., 2).

    Returns:
        1D NumPy array of shape [hidden_dim].
    """
    activation = {}

    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Capture last token vector from prompt sequence [1, seq_len, hidden_dim] -> [hidden_dim]
        activation["state"] = hidden_states[0, -1, :].detach().cpu().numpy()

    # 1. Attach hook to target layer
    target_layer = model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    # 2. Run model forward pass
    with torch.no_grad():
        _ = model(input_ids=input_ids, latent_plan=latent_plan)

    # 3. Clean up hook immediately to prevent memory leaks
    handle.remove()

    return activation["state"]
    
def run_controllability_benchmark():
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """Runs the latent controllability benchmark on the Blocksworld dataset."""
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
    tokenizer, gd_model = get_model_and_tokenizer(**model_config)
    gd_model.to(device)
    gd_model.eval()

    writer_model_path = f'{ds_name}_writer_model_{base_working_model}/writer_model.pt'
    model_config['writer_model_path'] = writer_model_path
    tokenizer, std_model = get_model_and_tokenizer(**model_config)
    std_model.to(device)
    std_model.eval()

    # model_config['working_model'] = 'gpt2_1024-s'
    # model_config['load_stage'] = 'base'
    # _, base_model = get_model_and_tokenizer(**model_config)
    # base_model.to(device)
    # base_model.eval()

    num_samples=500
    conditions = ["Adversarial_Swap"]
    
    
    num_layers = len(gd_model.layers)
    captured_activations_by_gd_model_layer = {i: [] for i in range(num_layers)}
    captured_activations_by_std_model_layer = {i: [] for i in range(num_layers)}

    captured_labels = []
    current_label = 0

    def make_hook_fn(layer_idx, model_type):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states.shape[1] > 1:
                last_token_act = hidden_states[:, -1, :].detach().cpu().numpy()
                if model_type == 'std':
                    captured_activations_by_std_model_layer[layer_idx].append(last_token_act)
                elif model_type == 'gd':
                    captured_activations_by_gd_model_layer[layer_idx].append(last_token_act)
                    if layer_idx == 0:
                        captured_labels.append(current_label)
        return hook_fn

    hook_handles = []
    for model, model_type in [(std_model, 'std'), (gd_model, 'gd')]:
        for idx, layer in enumerate(model.layers):
            handle = layer.register_forward_hook(make_hook_fn(idx, model_type))
            hook_handles.append(handle)
    acts_ctrl_a, acts_ctrl_b, acts_adv = [], [], []
    target_model = std_model
    target_layer = 3
    deltas = []
    ctrl_a_norms = []
    delta_norms = []
    for seed in [42]: #pylint: disable=too-many-nested-blocks
        set_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"\n=== RUNNING LATENT CONTROLLABILITY BENCHMARK ({num_samples} Samples {seed=}) ===")
        init_state = generate_random_state(num_shuffles=20, active_blocks=BLOCKS)
        for i in range(num_samples):
            init_str, goal_a, goal_b = generate_conflicting_goals(init_state=init_state)
            p_ctrl_a = f"[INIT] {init_str} [GOAL] {goal_a} "
            p_ctrl_b = f"[INIT] {init_str} [GOAL] {goal_b} "

            ids_a = tokenizer.encode(p_ctrl_a, return_tensors="pt").to(device)
            ids_b = tokenizer.encode(p_ctrl_b, return_tensors="pt").to(device)

            with torch.no_grad():
                mask_a = torch.ones_like(ids_a).to(device)
                mask_b = torch.ones_like(ids_b).to(device)

                p_lat_a = model.planner.get_initial_plan(ids_a, mask_a)
                p_lat_b = model.planner.get_initial_plan(ids_b, mask_b)


            h_a = extract_single_layer_state(
                target_model, ids_a, p_lat_a, target_layer
            )
            h_b = extract_single_layer_state(
                target_model, ids_b, p_lat_b, target_layer
            )
            h_swap = extract_single_layer_state(
                target_model, ids_a, p_lat_b, target_layer
            )

            acts_ctrl_a.append(h_a)
            acts_ctrl_b.append(h_b)
            acts_adv.append(h_swap)
            dh = h_swap - h_a
            deltas.append(dh)
            
            delta_norms.append(np.linalg.norm(dh))
            ctrl_a_norms.append(np.linalg.norm(h_a))

    deltas_tensor = torch.tensor(np.array(deltas))  # [N, hidden_dim]
    deltas_norm = F.normalize(deltas_tensor, p=2, dim=1)
    similarity_matrix = (
        torch.mm(deltas_norm, deltas_norm.T).cpu().numpy()
    )
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    pairwise_cosines = similarity_matrix[triu_indices]

    # 3. Print Steering Metrics
    mean_rel_mag = np.mean(np.array(delta_norms) / np.array(ctrl_a_norms)) * 100
    print(f"--- LAYER {target_layer} LATENT STEERING ANALYSIS ---")
    print(
        f"Mean Steering Magnitude relative to Base State: {mean_rel_mag:.2f}%"
    )
    print(f"Mean Pairwise Delta Alignment (Cosine Sim): {np.mean(pairwise_cosines):.4f}")

    # 4. Plot Alignment Histogram
    plt.figure(figsize=(6, 4), dpi=300)
    plt.hist(
        pairwise_cosines,
        bins=30,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.7,
    )
    plt.axvline(
        x=np.mean(pairwise_cosines),
        color="red",
        linestyle="--",
        label=f"Mean Sim = {np.mean(pairwise_cosines):.3f}",
    )
    plt.xlabel("Cosine Similarity between Latent Steering Vectors ($\Delta h$)", fontsize=10)
    plt.ylabel("Pair Count", fontsize=10)
    plt.title(f"Directional Consistency of Latent Injection (Layer {target_layer})", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("latent_steering_consistency_std_model.png")
    plt.close()

    # # Combine for PCA fitting
    # X_all = np.array(acts_ctrl_a + acts_ctrl_b + acts_adv)
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X_all)

    # n = len(acts_ctrl_a)
    # pca_a = X_2d[0:n]
    # pca_b = X_2d[n : 2 * n]
    # pca_swap = X_2d[2 * n :]

    # # 2. Plotting
    # plt.figure(figsize=(6, 6), dpi=300)
    # plt.scatter(
    #     pca_a[:, 0],
    #     pca_a[:, 1],
    #     alpha=0.6,
    #     label="Control A (Text A, Latent A)",
    #     color="blue",
    # )
    # plt.scatter(
    #     pca_b[:, 0],
    #     pca_b[:, 1],
    #     alpha=0.6,
    #     label="Control B (Text B, Latent B)",
    #     color="green",
    # )
    # plt.scatter(
    #     pca_swap[:, 0],
    #     pca_swap[:, 1],
    #     alpha=0.8,
    #     marker="X",
    #     s=80,
    #     label="Adversarial Swap (Text A, Latent B)",
    #     color="red",
    # )

    # plt.title(f"PCA of Last-Token Activations (Layer {target_layer})", fontsize=11)
    # plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
    # plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
    # plt.legend(loc="upper right", frameon=True)
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f"latent_controllability_pca_layer_{target_layer}.png", dpi=300)
    # plt.close()
    
if __name__ == "__main__":
    run_controllability_benchmark()
