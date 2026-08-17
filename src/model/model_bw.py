"""Blocksworld model components for latent planning and writing."""

import torch
from torch import nn
from torch.nn import functional as F

from .ar_gpt2 import GPT2Baseline, CausalSelfAttention, MLP

class CrossAttentionPlanUpdater(nn.Module):
    """
    The 'Navigator' mechanism.
    Uses Cross-Attention to dynamically ground the 
    Latent Plan (Query) against the actual physical actions taken so far (Keys/Values).
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, p_curr, past_actions):
        """
        P_curr: The current Latent Plan. Shape: [Batch, 1, Dim]
        past_actions: Sequence of action embeddings up to step t. Shape: [Batch, T, Dim]
        """
        if len(p_curr.shape) == 2:
            p_curr = p_curr.unsqueeze(1)

        q = self.ln_q(p_curr)
        k = v = self.ln_kv(past_actions)

        attn_out, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            need_weights=False
        )

        update = attn_out + self.mlp(self.ln_out(p_curr + attn_out))
        p_next = p_curr + + (self.gate * update)

        return p_next

class LatentPlanner(nn.Module):
    """
    Acts as the 'Compass'. Reads ONLY the prompt ([INIT] + [GOAL]) 
    and compresses it into a continuous static vector (z_g).
    """
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.encoder = GPT2Baseline(config)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.hidden_dim = config.n_embd

        self.init_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.action_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, latent_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(latent_dim, latent_dim)
        )

        self.updater = CrossAttentionPlanUpdater(hidden_dim=latent_dim)

    def get_initial_plan(self, prompt_ids, prompt_mask=None):
        """
        Computes the initial latent plan (P_0) from the prompt.
        """
        hidden_states = self.encoder(prompt_ids, return_hidden_states=True)

        if prompt_mask is not None:
            sequence_lengths = prompt_mask.sum(dim=1) - 1
            batch_size = prompt_ids.size(0)
            last_token_hidden = hidden_states[torch.arange(batch_size), sequence_lengths, :]
        else:
            last_token_hidden = hidden_states[:, -1, :]

        p_0 = self.init_projector(last_token_hidden).unsqueeze(1) # [Batch, 1, Dim]
        return p_0

    def step_plan(self, action_embedding, p_curr):
        """
        Updates the latent plan (p_curr) based on the latest action embedding."""
        lat_act = self.action_projector(action_embedding)
        if len(lat_act.shape) == 2:
            lat_act = lat_act.unsqueeze(1) # [Batch, 1, Dim]

        p_next = self.updater(p_curr, lat_act)
        return p_next

    def forward(self, prompt_ids, prompt_mask=None):
        """
        Computes the latent plan from the prompt."""
        hidden_states = self.encoder(prompt_ids, return_hidden_states=True)
        if prompt_mask is not None:
            sequence_lengths = prompt_mask.sum(dim=1) - 1
            batch_size = prompt_ids.size(0)
            last_token_hidden = hidden_states[torch.arange(batch_size), sequence_lengths, :]
        else:
            last_token_hidden = hidden_states[:, -1, :]

        latent_plan = self.init_projector(last_token_hidden)
        return latent_plan

class LatentDecoderBlock(nn.Module):
    """
    A standard GPT-2 block, but injected with a zero-initialized FiLM layer 
    that continuously broadcasts the static goal vector.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.film_scale = nn.Linear(config.n_embd, config.n_embd)
        self.film_shift = nn.Linear(config.n_embd, config.n_embd)

        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, latent_plan, film_mask=None):
        """
        x: The current sequence of embeddings. Shape: [Batch, T, Dim]
        latent_plan: The static latent plan vector. Shape: [Batch, 1, Dim]
        film_mask: Optional mask to control which tokens receive FiLM modulation. Shape: [Batch, T]
        """
        x = x + self.attn(self.ln_1(x))

        scale = self.film_scale(latent_plan)
        shift = self.film_shift(latent_plan)

        if len(scale.shape) == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        if film_mask is not None:
            film_mask_expanded = film_mask.unsqueeze(-1)
            scale = scale * film_mask_expanded
            shift = shift * film_mask_expanded

        x_modulated = (self.ln_2(x) * (1.0 + scale)) + shift
        x = x + self.mlp(x_modulated)

        return x

class LatentWriter(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """
    The unified End-to-End Continuous Global Conditioning Model.
    Ties the components together for simultaneous training and inference.
    """
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.planner = LatentPlanner(config, latent_dim=latent_dim)
        self.hidden_dim = config.n_embd

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)

        self.token_embedding.weight = self.planner.encoder.token_embedding.weight

        self.layers = nn.ModuleList([
            LatentDecoderBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_final = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, prompt_ids=None,
                prompt_mask=None, latent_plan=None,
                film_mask=None, position_ids=None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Computes the logits for the next token predictions given the input sequence."""
        device = input_ids.device
        b, t = input_ids.size()

        if latent_plan is None:
            assert prompt_ids is not None, "Must provide prompt_ids if latent_plan is not provided!"
            latent_plan = self.planner(prompt_ids, prompt_mask)

        if position_ids is None:
            position_ids = torch.arange(0, t, dtype=torch.long,
                                        device=device).unsqueeze(0).expand(b, t)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        for layer in self.layers:
            x = layer(x, latent_plan, film_mask=film_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, input_ids, comma_id, eos_id,
                 latent_plan=None,
                 max_new_tokens=50, temperature=0.0):
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """
        Standalone greedy generator to make testing the final model effortless.
        """
        self.eval()
        generated_ids = input_ids.clone()
        prompt_len = input_ids.size(1)

        prompt_mask = torch.ones_like(input_ids)

        if latent_plan is None:
            p_curr = self.planner.get_initial_plan(input_ids, prompt_mask)
        else:
            p_curr = latent_plan
        plan_history = p_curr.expand(-1, prompt_len, -1).clone()
        for _ in range(max_new_tokens):
            seq_len = generated_ids.size(1)
            if seq_len >= self.config.n_positions:
                break
            film_mask = (torch.arange(seq_len,
                                      device=input_ids.device
                                      ) >= prompt_len-1).float().unsqueeze(0).expand(
                                          generated_ids.size(0), -1)
            logits = self(
                input_ids=generated_ids,
                latent_plan=plan_history,
                film_mask=film_mask
            )
            next_token_logits = logits[:, -1, :]
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            token_val = next_token.item()
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            if token_val == comma_id:
                encoder_outputs = self.planner.encoder(generated_ids, return_hidden_states=True)
                act_emb = encoder_outputs[:, -1, :]
                p_curr = self.planner.step_plan(act_emb, p_curr)
            plan_history = torch.cat((plan_history, p_curr), dim=1)
            if token_val == eos_id:
                break
        return generated_ids
