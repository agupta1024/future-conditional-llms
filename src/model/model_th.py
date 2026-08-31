"""Treasure Hunt model components for latent planning and writing."""

from torch import nn

import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from .ar_gpt2 import GPT2Baseline

class FuturePredictor(nn.Module):
    """Compress the actual future text into a latent target vector."""

    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, hidden_dim),
        )

    def forward(self, past_features):
        """Project the pooled hidden state into the latent space."""
        return self.predictor(past_features)


class LatentPlanner(nn.Module):
    """Predict a latent plan from the prompt prefix."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, model_name_or_path="gpt2", latent_dim=2048,
                 custom_ar=False, config=None):
        super().__init__()
        print(f"Loading Base GPT-2 from {model_name_or_path}...")
        if custom_ar:
            self.encoder = GPT2Baseline(config_=config)
        else:
            self.encoder = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = self.encoder.config.n_embd
        self.latent_dim = latent_dim

        self.future_predictor = FuturePredictor(self.hidden_dim, self.latent_dim)
        self.custom_ar = custom_ar

    def mean_pooling(self, hidden_states, attention_mask):
        """Mean-pool token representations with the attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask  # (Batch Size, Hidden Dimension)

    def forward(self, input_ids, attention_mask,
                future_ids=None, future_mask=None):
        """
        past_ids: tokenized story (batch_size, seq_len)
        future_ids: tokenized story (batch_size, seq_len)
        attention_mask: attention mask (batch_size, seq_len)
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_states=True,
            )
            if isinstance(outputs, torch.Tensor):
                hidden_states = outputs
            else:
                hidden_states = outputs.hidden_states[-1]

            if future_ids is not None and future_mask is not None:
                true_future_hidden_states = self.encoder(
                    input_ids=future_ids,
                    attention_mask=future_mask,
                    return_hidden_states=True,
                )
                true_future_latent = self.mean_pooling(true_future_hidden_states, future_mask)
            else:
                true_future_hidden_states = None
                true_future_latent = None

        past_latent = self.mean_pooling(hidden_states, attention_mask)
        latent_plan = self.future_predictor(past_latent)

        return latent_plan, true_future_latent, true_future_hidden_states

    def get_initial_plan(self, input_ids, attention_mask):
        """Get the initial latent plan from the prompt."""
        with torch.no_grad():
            latent_plan, _, _ = self.forward(input_ids, attention_mask)
        return latent_plan

class LatentDecoderBlock(nn.Module): # pylint: disable=too-many-instance-attributes
    """Decoder block that can condition on a latent future plan."""

    def __init__(self, hidden_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)

        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, latent_plan, causal_mask, kv_cache=None, context_window=512):
        """
        x: The text generated so far (Batch, Seq_Len, Dim)
        latent_plan: The predicted future from the Planner (Batch, 1, Dim)
        causal_mask: Prevents looking at future tokens
        """
        norm_x = self.ln1(x)
        if kv_cache is not None:
            full_x = torch.cat([kv_cache, norm_x], dim=1)
        else:
            full_x = norm_x
        if context_window is not None and full_x.size(1) > context_window:
            full_x = full_x[:, -context_window:, :]
        new_kv_cache = full_x
        active_mask = causal_mask if x.size(1) > 1 else None
        attn_out, _ = self.self_attn(
            query=norm_x,
            key=full_x,
            value=full_x,
            attn_mask=active_mask,
            need_weights=False,
        )
        x = x + attn_out

        scale = self.film_scale(latent_plan)  # (Batch, 1, Dim)
        shift = self.film_shift(latent_plan)  # (Batch, 1, Dim)
        if len(scale.shape) == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        x_modulated = (self.ln3(x) * (1.0 + scale)) + shift
        x = x + self.mlp(x_modulated)
        return x, new_kv_cache


class LatentWriter(nn.Module):
    """Decode text while conditioning on a latent future plan."""

    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

    def __init__(self, base_model_path="gpt2", vocab_size=50257,
                 max_seq_len=1024, latent_dim=512,
                 custom_ar=False, config=None):
        super().__init__()
        self.planner = LatentPlanner(
            model_name_or_path=base_model_path,
            latent_dim=latent_dim,
            custom_ar=custom_ar,
            config=config,
        )
        self.config = config
        # for param in self.planner.parameters():
        #     param.requires_grad = False
        self.hidden_dim = self.planner.hidden_dim
        pretrained_embeddings = self.planner.encoder.token_embedding.weight.data

        self.token_embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.token_embedding.weight.data.copy_(pretrained_embeddings)

        self.position_embedding = nn.Embedding(max_seq_len, self.hidden_dim)

        num_layers = self.planner.encoder.config.n_layer
        num_heads = self.planner.encoder.config.n_head
        self.layers = nn.ModuleList([
            LatentDecoderBlock(hidden_dim=self.hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def generate_causal_mask(self, seq_len, device):
        """Build an upper-triangular causal mask for autoregressive decoding."""
        return torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1).to(device)

    def forward(self, input_ids, latent_plan=None,
                prompt_ids=None, prompt_mask=None,
                position_ids=None, kv_caches=None, use_cache=False,
                context_window=512, return_hidden_states=False):
        #pylint: disable=too-many-locals
        """
        input_ids: (Batch, Seq_Len)
        latent_plan: (Batch, Dim) -> We will unsqueeze it to (Batch, 1, Dim)
        """
        _batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if latent_plan is None:
            assert prompt_ids is not None, "Must provide prompt_ids if latent_plan is None!"
            latent_plan, _, _ = self.planner(prompt_ids, prompt_mask)

        if position_ids is None:
            past_seq_len = kv_caches[0].size(1) if kv_caches is not None else 0
            position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, dtype=torch.long,
                                        device=device).unsqueeze(0).expand(_batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        new_kv_caches = [] if use_cache else None
        causal_mask = self.generate_causal_mask(seq_len, device)
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_layer_cache = layer(
                x,
                latent_plan,
                causal_mask,
                kv_cache=layer_cache,
                context_window=context_window
            )
            if use_cache:
                new_kv_caches.append(new_layer_cache)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        if return_hidden_states:
            return x
        if use_cache:
            return logits, new_kv_caches
        return logits

    @torch.no_grad()
    def generate(self, input_ids, eos_token_id, pad_token_id=None,
                    latent_plan=None, context_window=512,
                    max_new_tokens=50, temperature=0.0, **_kwargs):
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """
        Standalone greedy generator to make testing the final model effortless.
        """
        self.eval()
        generated_ids = input_ids.clone()
        batch_size, _ = input_ids.shape
        prompt_mask = torch.ones_like(input_ids)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool,
                                          device=input_ids.device)
        if latent_plan is None:
            latent_plan = self.planner.get_initial_plan(input_ids, prompt_mask)

        fill_pad_id = (pad_token_id if pad_token_id is not None
                       else (eos_token_id if eos_token_id is not None else 0))

        writer_kv_caches = None
        current_input_ids = input_ids[:, -context_window:]
        for _ in range(max_new_tokens):
            seq_len = current_input_ids.size(1)
            if seq_len >= self.config.n_positions:
                break

            logits, writer_kv_caches = self(
                input_ids=current_input_ids,
                latent_plan=latent_plan,
                kv_caches=writer_kv_caches,
                use_cache=True,
                context_window=context_window
            )
            next_token_logits = logits[:, -1, :]
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                next_token_flat = next_token.squeeze(-1)
                is_eos = next_token_flat == eos_token_id
                next_token_flat = torch.where(
                    unfinished_sequences,
                    next_token_flat,
                    fill_pad_id
                )
                unfinished_sequences = unfinished_sequences & (~is_eos)
                next_token = next_token_flat.unsqueeze(-1)

            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            if eos_token_id is not None and not unfinished_sequences.any():
                break
            current_input_ids = next_token
        return generated_ids
