import torch

# 1. Load weights from both checkpoints
sft_state = torch.load("././blocksworld_writer_model_gpt2_1024/writer_model.pt", map_location="cpu")
eggroll_state = torch.load("./blocksworld_writer_model_gpt2_1024_EGGROLL/writer_model.pt", map_location="cpu")

# 2. Measure parameter distance across layers
diffs = []
for key in sft_state.keys():
    if "weight" in key and key in eggroll_state:
        diff = torch.norm(eggroll_state[key].float() - sft_state[key].float()).item()
        diffs.append(diff)
        if diff > 0:
            print(f"Layer {key} | Weight Diff Norm: {diff:.6f}")

print(f"\nAverage Layer Difference: {sum(diffs) / len(diffs):.6f}")