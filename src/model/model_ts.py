"""TinyStories model components for latent planning and writing."""

from torch import nn

import torch
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
                 custom_ar=False, model_config=None):
        super().__init__()
        print(f"Loading Base GPT-2 from {model_name_or_path}...")
        if custom_ar:
            self.encoder = GPT2Baseline(config_=model_config)
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
        split_idx: integer defining where the "past" ends and "future" begins
        """
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
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
                # true_future_latent = self.mean_pooling(true_future_hidden_states, future_mask)
                true_future_latent = true_future_hidden_states[torch.arange(batch_size),
                                                               sequence_lengths, :]
            else:
                true_future_hidden_states = None
                true_future_latent = None

        # pooled_past_latent = self.mean_pooling(hidden_states, attention_mask)
        last_token_hidden = hidden_states[torch.arange(batch_size), sequence_lengths, :]
        # --- Predict the Future ---
        # Only the predictor tracks gradients
        latent_plan = self.future_predictor(last_token_hidden)

        return latent_plan, true_future_latent, true_future_hidden_states


class LatentDecoderBlock(nn.Module): # pylint: disable=too-many-instance-attributes
    """Decoder block that can condition on a latent future plan."""

    def __init__(self, hidden_dim=512, num_heads=4, dropout=0.1, film=False):
        super().__init__()
        self.film = film
        # 1. Causal Self-Attention (Looking at the words written so far)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden_dim)

        if film:
            # 2. FiLM Layers (Feature-wise Linear Modulation)
            # Injects the latent plan directly into the activation stream
            self.film_scale = nn.Linear(hidden_dim, hidden_dim)
            self.film_shift = nn.Linear(hidden_dim, hidden_dim)

            nn.init.zeros_(self.film_scale.weight)
            nn.init.zeros_(self.film_scale.bias)
            nn.init.zeros_(self.film_shift.weight)
            nn.init.zeros_(self.film_shift.bias)
        else:
            # 2. Cross-Attention (Looking at the frozen Latent Plan)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ln2 = nn.LayerNorm(hidden_dim)

        # 3. Feed Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, latent_plan, causal_mask):
        """
        x: The text generated so far (Batch, Seq_Len, Dim)
        latent_plan: The predicted future from the Planner (Batch, 1, Dim)
        causal_mask: Prevents looking at future tokens
        """
        attn_out, _ = self.self_attn(
            query=self.ln1(x),
            key=self.ln1(x),
            value=self.ln1(x),
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out  # Residual connection

        if self.film:
            scale = self.film_scale(latent_plan)  # (Batch, 1, Dim)
            shift = self.film_shift(latent_plan)  # (Batch, 1, Dim)
            if len(scale.shape) == 2:
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)
            # Modulate the normalized text features
            x_modulated = (self.ln3(x) * (1.0 + scale)) + shift
        else:
            # --- Step 2: Cross Attention (The "Compass") ---
            # Query comes from text (x). Keys/Values come from the latent_plan.
            # No mask needed here because the latent_plan is timeless.
            cross_out, _ = self.cross_attn(
                query=self.ln2(x),
                key=latent_plan,
                value=latent_plan,
                need_weights=False,
            )
            x_modulated = x + cross_out  # Residual connection
            x_modulated = self.ln3(x_modulated)

        x = x + self.mlp(x_modulated)
        return x


class LatentWriter(nn.Module):
    """Decode text while conditioning on a latent future plan."""

    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

    def __init__(self, base_model_path="gpt2", vocab_size=50257,
                 max_seq_len=1024, latent_dim=512,
                 custom_ar=False, model_config=None, film=False):
        super().__init__()
        self.planner = LatentPlanner(
            model_name_or_path=base_model_path,
            latent_dim=latent_dim,
            custom_ar=custom_ar,
            model_config=model_config,
        )
        for param in self.planner.parameters():
            param.requires_grad = False
        self.hidden_dim = self.planner.hidden_dim
        pretrained_embeddings = self.planner.encoder.token_embedding.weight.data

        # Token and Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.token_embedding.weight.data.copy_(pretrained_embeddings)

        self.position_embedding = nn.Embedding(max_seq_len, self.hidden_dim)
        # Stack of Custom Decoder Blocks
        num_layers = self.planner.encoder.config.n_layer
        num_heads = self.planner.encoder.config.n_head
        self.layers = nn.ModuleList([
            LatentDecoderBlock(hidden_dim=self.hidden_dim, num_heads=num_heads, film=film)
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
                return_hidden_states=False):
        """
        input_ids: (Batch, Seq_Len)
        latent_plan: (Batch, Dim) -> We will unsqueeze it to (Batch, 1, Dim)
        """
        _batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if latent_plan is None:
            assert prompt_ids is not None, "Must provide prompt_ids if latent_plan is None!"
            latent_plan, _, _ = self.planner(prompt_ids, prompt_mask)

        # 1. Embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 2. Causal Mask
        causal_mask = self.generate_causal_mask(seq_len, device)

        # 3. Pass through Decoder Blocks
        for layer in self.layers:
            x = layer(x, latent_plan, causal_mask)

        x = self.ln_final(x)
        if return_hidden_states:
            return x
        logits = self.lm_head(x)
        return logits
