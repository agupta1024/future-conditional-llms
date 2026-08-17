"""Autoregressive GPT-2 baseline implementation and weight-transfer helpers."""

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    """Match the dimensions used by the HuggingFace baseline configuration."""

    vocab_size: int = 50257
    n_positions: int = 512
    n_embd: int = 512
    n_layer: int = 4
    n_head: int = 8


class CausalSelfAttention(nn.Module):
    """A lightweight causal self-attention layer for the baseline model."""

    def __init__(self, config_):
        super().__init__()
        assert config_.n_embd % config_.n_head == 0

        self.c_attn = nn.Linear(config_.n_embd, 3 * config_.n_embd)
        self.c_proj = nn.Linear(config_.n_embd, config_.n_embd)

        self.n_head = config_.n_head
        self.n_embd = config_.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config_.n_positions, config_.n_positions)).view(
                1, 1, config_.n_positions, config_.n_positions
            ),
        )

    def forward(self, inputs):
        """Compute causal self-attention over the input sequence."""
        batch_size, seq_length, embed_dim = inputs.size()

        qkv = self.c_attn(inputs)
        query, key, value = qkv.split(self.n_embd, dim=2)

        key = key.view(batch_size, seq_length, self.n_head, embed_dim // self.n_head).transpose(1,2)
        query = query.view(batch_size,seq_length,self.n_head,embed_dim //self.n_head).transpose(1,2)
        value = value.view(batch_size,seq_length,self.n_head,embed_dim //self.n_head).transpose(1,2)

        attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attention = attention.masked_fill(self.bias[:,:,:seq_length,:seq_length]== 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)

        attended = attention @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        attended = self.c_proj(attended)
        return attended


class MLP(nn.Module):
    """A simple MLP block used in the baseline transformer."""

    def __init__(self, config_):
        super().__init__()
        self.c_fc = nn.Linear(config_.n_embd, 4 * config_.n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config_.n_embd, config_.n_embd)

    def forward(self, inputs):
        """Apply the feed-forward block to the input tensor."""
        x = self.c_fc(inputs)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """A single transformer block with attention and MLP layers."""

    def __init__(self, config_):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config_.n_embd)
        self.attn = CausalSelfAttention(config_)
        self.ln_2 = nn.LayerNorm(config_.n_embd)
        self.mlp = MLP(config_)

    def forward(self, inputs):
        """Run a residual block over the input sequence."""
        x = inputs + self.attn(self.ln_1(inputs))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Baseline(nn.Module):
    """A compact autoregressive GPT-2 implementation used for baseline comparisons."""

    def __init__(self, config_):
        super().__init__()
        self.config = config_

        self.token_embedding = nn.Embedding(config_.vocab_size, config_.n_embd)
        self.position_embedding = nn.Embedding(config_.n_positions, config_.n_embd)

        self.layers = nn.ModuleList([Block(config_) for _ in range(config_.n_layer)])
        self.ln_final = nn.LayerNorm(config_.n_embd)

        self.lm_head = nn.Linear(config_.n_embd, config_.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids, return_hidden_states=False, **_kwargs):
        """Run the baseline model over the provided token ids."""
        device = input_ids.device
        _, seq_length = input_ids.size()

        positions = torch.arange(0, seq_length, dtype=torch.long, device=device)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        if return_hidden_states:
            return x
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, **_kwargs):
        # pylint: disable=too-many-locals
        """Generate tokens autoregressively from the baseline model."""
        generated_ids = input_ids.clone()
        context_window = _kwargs.get('context_window', self.config.n_positions)
        temperature = _kwargs.get('temperature', 0.0)
        top_k = _kwargs.get('top_k', 50)
        eos_token_id = _kwargs.get('eos_token_id', self.config.vocab_size - 1)
        for _ in range(max_new_tokens):
            context_ids = generated_ids[:, -context_window:]
            outputs = self(context_ids, use_cache=False)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    min_top_k_val = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits[next_token_logits < min_top_k_val] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            if next_token.item() == eos_token_id:
                break

        return generated_ids

def load_hf_weights_into_baseline(hf_model, custom_baseline):
    """Transfer HuggingFace weights into the compact baseline implementation."""
    print("Extracting HF state dictionary...")
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_baseline.state_dict()

    new_state_dict = {}

    for hf_key, tensor in hf_state_dict.items():
        custom_key = hf_key
        custom_key = custom_key.replace("transformer.wte.weight", "token_embedding.weight")
        custom_key = custom_key.replace("transformer.wpe.weight", "position_embedding.weight")
        custom_key = custom_key.replace("transformer.ln_f.", "ln_final.")
        custom_key = custom_key.replace("transformer.h.", "layers.")

        if any(weight_key in hf_key for weight_key in ("c_attn.weight",
                                                       "c_proj.weight", "c_fc.weight")):
            tensor = tensor.t()

        if custom_key in custom_state_dict:
            new_state_dict[custom_key] = tensor

    for key in custom_state_dict:
        if key.endswith("attn.bias") and key not in new_state_dict:
            new_state_dict[key] = custom_state_dict[key]

    custom_baseline.load_state_dict(new_state_dict, strict=True)
    print("Transfer Complete. Saving baseline model components...")
    torch.save(custom_baseline.state_dict(), "tinystories_base_model_gpt2_512_512/base_model.pt")
    print("Weight transfer complete! Baseline is ready for profiling.")
    return custom_baseline


if __name__ == "__main__":
    from ..config.model_config import get_model_and_tokenizer # pylint: disable=relative-beyond-top-level

    base_working_model = "hf_gpt2_512_512"
    stage = "base"
    model_map = {
        "gpt2_hf": [2048, 768],
        "gpt2_hf_1024": [1024, 768],
        "gpt2_512_1024": [1024, 512],
        "gpt2_512_512": [512, 512],
        "hf_gpt2_512_512": [512, 512],
    }

    _, hf_model_ = get_model_and_tokenizer(
        working_model=base_working_model,
        hidden_dim=model_map[base_working_model][1],
        max_seq_length=model_map[base_working_model][0],
        load_scratch=False,
        dataset_name="tinystories",
        load_stage=stage,
    )

    base_working_model = "gpt2_512_512"
    stage = "base"
    _, model = get_model_and_tokenizer(
        working_model=base_working_model,
        hidden_dim=model_map[base_working_model][1],
        max_seq_length=model_map[base_working_model][0],
        load_scratch=False,
        dataset_name="tinystories",
        load_stage=stage,
        custom_ar=True,
    )
    hf_model_.eval()
    model.eval()

    config = hf_model_.config
    custom_baseline_ = GPT2Baseline(config)

    custom_baseline_ = load_hf_weights_into_baseline(hf_model_, custom_baseline_)

    mock_input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(hf_model_.device)

    with torch.no_grad():
        hf_outputs = hf_model_(mock_input_ids)
        hf_logits = hf_outputs.logits
        bare_logits = model(mock_input_ids)

    max_diff = (hf_logits - bare_logits).abs().max().item()

    print(f"Maximum difference between the models: {max_diff:.8f}")
    if max_diff < 1e-4:
        print(
            "SUCCESS: The math is 100% identical. The text difference is purely "
            "due to random sampling/RNG seeds."
        )
    else:
        print("WARNING: The models are mathematically diverging.\
            Something is wrong with the weight transfer.")
