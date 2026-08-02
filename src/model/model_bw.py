"""Blocksworld model components for latent planning and writing."""

from torch import nn

import torch
from transformers import GPT2LMHeadModel

from .ar_gpt2 import GPT2Baseline

class FuturePredictor(nn.Module):
    """Compress past context into a target future vector."""

    def __init__(self, hidden_dim, latent_dim, dropout_rate=0.2):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, hidden_dim)
        )

    def forward(self, past_features):
        """Project pooled hidden states into the latent space."""
        return self.predictor(past_features)

class LatentPlanner(nn.Module):
    """Read the prompt and compress it into a continuous latent plan."""

    def __init__(self, model_name_or_path='gpt2', latent_dim=512, custom_ar=False, config=None):
        super().__init__()
        if custom_ar:
            self.encoder = GPT2Baseline(config_=config)
        else:
            self.encoder = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = self.encoder.config.n_embd
        self.future_predictor = FuturePredictor(self.hidden_dim, latent_dim)

    def mean_pooling(self, hidden_states, attention_mask):
        """Mean-pool hidden states using the attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask,
                    future_ids=None, future_mask=None):
        """Predict the latent future plan from prompt tokens."""
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

class LatentDecoderBlock(nn.Module):
    """Blend local decoding with a latent future plan via FiLM."""

    def __init__(self, hidden_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        # 1. Local Syntax Engine (Self-Attention)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden_dim)

        # 2. Continuous Global Conditioning (FiLM)
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)

        # Zero-init so it starts as an Identity function (protects pre-trained grammar)
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

        # 3. Feed Forward
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        # self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, latent_plan, causal_mask):
        """Apply causal self-attention, FiLM conditioning, and an MLP."""
        # Step 1: Look at the local text
        attn_out, _ = self.self_attn(
            query=self.ln1(x),
            key=self.ln1(x),
            value=self.ln1(x),
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Step 2: Inject the Global Goal via FiLM
        scale = self.film_scale(latent_plan)
        shift = self.film_shift(latent_plan)

        # Safely ensure they are exactly 3D [Batch, 1, Dim] for broadcasting
        if len(scale.shape) == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        x_modulated = (self.ln3(x) * (1.0 + scale)) + shift

        # Step 3: MLP
        x = x + self.mlp(x_modulated)
        return x

class LatentWriter(nn.Module):
    """Decode trajectories while conditioning on a latent plan."""
    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

    def __init__(self, base_model_path='gpt2', vocab_size=50257, max_seq_len=1024,
                 latent_dim=512, custom_ar=False, config=None):
        super().__init__()

        self.planner = LatentPlanner(
            model_name_or_path=base_model_path,
            latent_dim=latent_dim,
            custom_ar=custom_ar,
            config=config,
        )
        self.hidden_dim = self.planner.hidden_dim

        # Standard GPT-2 Embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, self.hidden_dim)

        pretrained_embeddings = self.planner.encoder.token_embedding.weight.data
        self.token_embedding.weight.data.copy_(pretrained_embeddings)

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
        """Build a causal attention mask for autoregressive decoding."""
        return torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1).to(device)

    def forward(self, input_ids, prompt_ids=None, prompt_mask=None, latent_plan=None):
        """
        input_ids: The sequence to decode (e.g., Trajectory).
        prompt_ids: Passed during training to generate the plan.
        latent_plan: Passed during inference to avoid recalculating the plan.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if latent_plan is None:
            assert prompt_ids is not None, "Must provide prompt_ids if latent_plan is None!"
            latent_plan, _, _ = self.planner(prompt_ids, prompt_mask)

        positions = torch.arange(0, seq_len, dtype=torch.long,
                                 device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = self.generate_causal_mask(seq_len, device)
        for layer in self.layers:
            x = layer(x, latent_plan, causal_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
