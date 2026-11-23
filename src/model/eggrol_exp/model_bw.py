"""Blocksworld model components for latent planning and writing."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .ar_gpt2 import GPT2Baseline, CausalSelfAttention, MLP

class EggrollAdapter(nn.Module):
    """
    A Low-Rank Adapter designed strictly for Evolution Strategies.
    Requires NO gradients.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.in_dim = base_layer.in_features
        self.out_dim = base_layer.out_features
        
        # 2. EGGROLL Magic: We accumulate into a FULL-RANK matrix
        # This requires 0 gradient overhead, just standard VRAM storage.
        self.delta_W = nn.Parameter(torch.zeros_like(self.base_layer.weight), requires_grad=False)
        
        self.A_noise = None
        self.B_noise = None
        self.active_noise_scale = 0.0
    
    @property
    def weight(self):
        """Dynamically constructs the effective weight matrix for C++ kernels."""
        # 1. Base weights + accumulated full-rank EGGROLL updates
        effective_w = self.base_layer.weight + self.delta_W
        
        # 2. Add the temporary evolutionary noise if currently injected (Phase 1)
        if self.A_noise is not None:
            # Reconstruct the full-rank noise matrix dynamically
            # A_noise: [out_dim, rank], B_noise: [in_dim, rank] -> Matrix: [out_dim, in_dim]
            noise_matrix = self.A_noise @ self.B_noise.T
            effective_w = effective_w + (self.active_noise_scale * noise_matrix)
            
        return effective_w

    @property
    def bias(self):
        return self.base_layer.bias
        
    @property
    def in_features(self):
        return self.base_layer.in_features
        
    @property
    def out_features(self):
        return self.base_layer.out_features

    def inject_noise(self, generator: torch.Generator, noise_scale: float):
        """Generates a temporary mutation using a specific random seed."""
        # A ~ N(0,1) with shape [out_dim, r]
        # B ~ N(0,1) with shape [in_dim, r]
        self.A_noise = torch.randn(self.out_dim, self.rank, generator=generator, device=self.delta_W.device)
        self.B_noise = torch.randn(self.in_dim, self.rank, generator=generator, device=self.delta_W.device)
        
        # Paper Scaling: E = (sigma / sqrt(r)) * (A @ B^T)
        self.active_noise_scale = noise_scale / math.sqrt(self.rank)

    def accumulate_update(self, generator: torch.Generator, step_size: float,
                          weight_decay: float = 1e-3, max_norm: float = 1.0):
        """Reconstructs the full-rank noise matrix and adds it to the central weights."""
        if weight_decay > 0:
            self.delta_W.data.mul_(1.0 - weight_decay)
            
        A_n = torch.randn(self.out_dim, self.rank, generator=generator, device=self.delta_W.device)
        B_n = torch.randn(self.in_dim, self.rank, generator=generator, device=self.delta_W.device)
        
        # Update Rule: Theta = Theta + lr * fitness_scalar * (A @ B^T) / sqrt(rank)
        # We instantiate the full-rank matrix here just for the update addition, then drop it.
        update_matrix = A_n @ B_n.T
        dim_scale = math.sqrt(self.in_features * self.out_features * self.rank)
        update_matrix = update_matrix / dim_scale
        self.delta_W.data += step_size * update_matrix
        # self.delta_W.data += (step_size / math.sqrt(self.rank)) * update_matrix

        current_norm = torch.norm(self.delta_W.data)
        if current_norm > max_norm:
            self.delta_W.data.mul_(max_norm / current_norm)

    def get_delta_norm(self):
        """Helper to monitor weight magnitude."""
        return torch.norm(self.delta_W.data).item()

    def clear_noise(self):
        """Reverts back to the central mean weights (used for final evaluation)."""
        self.A_noise = None
        self.B_noise = None
        self.active_noise_scale = 0.0

    def forward(self, x):
        # 1. Base network forward + our learned full-rank updates
        base_out = self.base_layer(x) + F.linear(x, self.delta_W)
        
        # 2. If noise is injected (during Phase 1), apply it efficiently
        if self.A_noise is not None:
            # Mathematical identity: x @ (A @ B^T)^T = (x @ B) @ A^T
            # This avoids creating the [out_dim, in_dim] noise matrix entirely!
            low_rank_out = (x @ self.B_noise) @ self.A_noise.T
            return base_out + (self.active_noise_scale * low_rank_out)
            
        return base_out

def inject_eggroll_adapters(model, rank=16):
    """
    Utility to recursively replace targeted nn.Linear layers with EggrollAdapters.
    You should target the Planner Encoder, FiLM projections, and Decoder MLPs.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace layer with adapter
            setattr(model, name, EggrollAdapter(module, rank=rank))
        else:
            inject_eggroll_adapters(module, rank=rank)
    return model

def remove_eggroll_adapters_and_save(model, save_path):
    """
    Fuses the EGGROLL delta_W back into the base weights, strips the adapters, 
    and saves a pristine standard Hugging Face model.
    """
    def _unwrap(module):
        for name, child in module.named_children():
            if isinstance(child, EggrollAdapter):
                norm = torch.norm(child.delta_W.data).item()
                print(f"Layer: {name} | Delta Norm: {norm:.6f}")
                # Merge the learned full-rank updates permanently
                child.base_layer.weight.data += child.delta_W.data
                # Replace adapter with the original linear layer
                setattr(module, name, child.base_layer)
            else:
                _unwrap(child)
                
    print(f"Fusing EGGROLL weights and saving model to {save_path}...")
    _unwrap(model)
    model.save_pretrained(save_path)
    print("Save complete!")
    
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

    def get_initial_plan(self, prompt_ids, prompt_mask=None,
                         position_ids=None, use_cache=False,
                         context_window=1024):
        """
        Computes the initial latent plan (P_0) from the prompt.
        """
        if use_cache:
            hidden_states, kv_caches = self.encoder(
                prompt_ids, attention_mask=prompt_mask,
                return_hidden_states=True, use_cache=True,
                context_window=context_window
            )
        else:
            hidden_states = self.encoder(prompt_ids, attention_mask=prompt_mask,
                                         return_hidden_states=True)
            kv_caches = None

        if prompt_mask is not None:
            sequence_lengths = (prompt_mask != 0).cumsum(dim=1).argmax(dim=1)
            batch_size = prompt_ids.size(0)
            last_token_hidden = hidden_states[torch.arange(batch_size), sequence_lengths, :]
        else:
            last_token_hidden = hidden_states[:, -1, :]

        p_0 = self.init_projector(last_token_hidden).unsqueeze(1)
        if use_cache:
            return p_0, kv_caches
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
        hidden_states = self.encoder(prompt_ids, attention_mask=prompt_mask,
                                    return_hidden_states=True)
        if prompt_mask is not None:
            sequence_lengths = (prompt_mask != 0).cumsum(dim=1).argmax(dim=1)
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

    def forward(self, x, attention_mask, latent_plan, film_mask=None, kv_cache=None, context_window=1024):
        """
        x: The current sequence of embeddings. Shape: [Batch, T, Dim]
        latent_plan: The static latent plan vector. Shape: [Batch, 1, Dim]
        film_mask: Optional mask to control which tokens receive FiLM modulation. Shape: [Batch, T]
        """
        attn_out, new_kv_cache = self.attn(
            self.ln_1(x),
            kv_cache=kv_cache,
            context_window=context_window,
            attention_mask=attention_mask
        )
        x = x + attn_out

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

        return x, new_kv_cache

class LatentWriter(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """
    The unified End-to-End Continuous Global Conditioning Model.
    Ties the components together for simultaneous training and inference.
    """
    def __init__(self, config, latent_dim=512, **_kwargs):
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

    def forward(self, input_ids, attention_mask=None,
                prompt_ids=None,
                prompt_mask=None, latent_plan=None,
                film_mask=None, position_ids=None,
                kv_caches=None, use_cache=False,
                context_window=1024, **_kwargs):
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """Computes the logits for the next token predictions given the input sequence."""
        device = input_ids.device
        b, t = input_ids.size()

        if latent_plan is None:
            assert prompt_ids is not None, "Must provide prompt_ids if latent_plan is not provided!"
            latent_plan = self.planner(prompt_ids, prompt_mask)

        if position_ids is None:
            past_seq_len = kv_caches[0][0].size(2) if kv_caches is not None else 0
            position_ids = torch.arange(past_seq_len, past_seq_len + t, dtype=torch.long,
                                        device=device).unsqueeze(0).expand(b, t)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        new_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_layer_cache = layer(
                x,
                attention_mask,
                latent_plan,
                film_mask=film_mask,
                kv_cache=layer_cache,
                context_window=context_window
            )
            if use_cache:
                new_kv_caches.append(new_layer_cache)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kv_caches
        return logits

    @torch.no_grad()
    def generate(self, input_ids, comma_id, eos_token_id,
                 attention_mask=None, latent_plan=None, context_window=1024,
                 max_new_tokens=50, temperature=0.0,
                 **_kwargs):
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """
        Standalone greedy generator to make testing the final model effortless.
        """
        self.eval()
        b_size = input_ids.size(0)
        generated_ids = input_ids.clone()

        prompt_mask = torch.ones_like(input_ids)
        if attention_mask is not None:
            prompt_mask = attention_mask

        if latent_plan is None:
            p_curr, planner_kv_caches = self.planner.get_initial_plan(
                input_ids, prompt_mask, use_cache=True, context_window=context_window
            )
        else:
            p_curr = latent_plan
            _, planner_kv_caches = self.planner.encoder(
                input_ids, use_cache=True, context_window=context_window
            )
        # plan_history = p_curr.expand(-1, prompt_len, -1).clone()

        writer_kv_caches = None
        current_input_ids = input_ids[:, -context_window:]
        prompt_len = current_input_ids.size(1)
        current_plan = p_curr.expand(-1, prompt_len, -1)
        current_film_mask = (torch.arange(prompt_len, device=input_ids.device) >=
                             prompt_len - 1).float().unsqueeze(0).expand(generated_ids.size(0), -1)

        encoder_unprocessed_tokens = []
        for _ in range(max_new_tokens):
            seq_len = current_input_ids.size(1)
            if seq_len >= self.config.n_positions:
                break

            logits, writer_kv_caches = self(
                input_ids=current_input_ids,
                latent_plan=current_plan,
                film_mask=current_film_mask,
                kv_caches=writer_kv_caches,
                use_cache=True,
                context_window=context_window,
                # attention_mask=attention_mask
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

            encoder_unprocessed_tokens.append(next_token)
            if token_val == comma_id:
                chunk_ids = torch.cat(encoder_unprocessed_tokens, dim=1)
                encoder_outputs, planner_kv_caches = self.planner.encoder(
                    chunk_ids,
                    return_hidden_states=True,
                    use_cache=True,
                    kv_caches=planner_kv_caches,
                    context_window=context_window
                )
                act_emb = encoder_outputs[:, -1, :]
                p_curr = self.planner.step_plan(act_emb, p_curr)
                encoder_unprocessed_tokens.clear()

            # # plan_history = torch.cat((plan_history, p_curr), dim=1)
            if token_val == eos_token_id:
                break
            current_input_ids = next_token
            current_plan = p_curr
            current_film_mask = torch.ones((generated_ids.size(0), 1), device=input_ids.device)
            # Extend attention mask
            # new_attn_mask = torch.ones((b_size, 1), dtype=torch.long, device=input_ids.device)
            # attention_mask = torch.cat([attention_mask, new_attn_mask], dim=1)
        return generated_ids

    @torch.no_grad()
    def generate_batched(self, input_ids, comma_id, eos_token_id,
                 attention_mask=None, latent_plan=None, context_window=1024,
                 max_new_tokens=50, temperature=0.0, pad_token_id=0,
                 **_kwargs):
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """
        Batched generator optimized for pure inference and EGGROLL evolution.
        Supports left-padded batched inputs safely.
        """
        self.eval()
        b_size = input_ids.size(0)
        device = input_ids.device

        generated_ids = input_ids.clone()
        
        # 1. Handle attention_mask for left-padding
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        # 2. Initialize Planner 
        if latent_plan is None:
            # Ensure the planner ignores [PAD] tokens in the prompt
            p_curr, _ = self.planner.get_initial_plan(
                input_ids, attention_mask, position_ids,
                use_cache=True,
                context_window=context_window
            )
        else:
            p_curr = latent_plan

        writer_kv_caches = None
        current_input_ids = input_ids[:, -context_window:]
        prompt_len = current_input_ids.size(1)
        
        # 3. Setup Position IDs for correct positional alignment on left-padded batches
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        curr_pos_ids = position_ids[:, -prompt_len:]
        
        current_plan = p_curr.expand(-1, prompt_len, -1)
        
        # FiLM mask: 0.0 for prompt tokens, 1.0 ONLY on the final prompt boundary token
        current_film_mask = torch.zeros_like(current_input_ids, dtype=torch.float)
        current_film_mask[:, -1] = 1.0

        # Track which sequences in the batch have hit the EOS token
        unfinished_sequences = torch.ones(b_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            seq_len = current_input_ids.size(1)
            if seq_len >= self.config.n_positions:
                break
            
            # Forward pass using KV caching for the heavy causal decoder
            logits, writer_kv_caches = self(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                latent_plan=current_plan,
                film_mask=current_film_mask,
                position_ids=curr_pos_ids,
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
                
            # 4. Vectorized EOS handling (freeze sequences that finished)
            next_token = next_token * unfinished_sequences.unsqueeze(1).long() + \
                         pad_token_id * (~unfinished_sequences).unsqueeze(1).long()
                         
            has_hit_eos = (next_token.squeeze(-1) == eos_token_id)
            unfinished_sequences = unfinished_sequences & ~has_hit_eos

            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            # 5. Vectorized Trigger Condition (Batch-safe Planner Update)
            is_comma = (next_token.squeeze(-1) == comma_id)
            
            if is_comma.any():
                # Re-encode the full sequence to avoid batch chunking misalignment.
                # (Decoder is still KV-cached, so this small encoder forward pass is extremely fast)
                planner_attn_mask = (generated_ids != pad_token_id).long()
                
                encoder_outputs = self.planner.encoder(
                    generated_ids,
                    attention_mask=planner_attn_mask,
                    return_hidden_states=True,
                    use_cache=False,  
                    context_window=context_window
                )
                last_idx = (planner_attn_mask != 0).cumsum(dim=1).argmax(dim=1)
                act_emb = encoder_outputs[torch.arange(b_size), last_idx, :].unsqueeze(1)
                # act_emb = encoder_outputs[:, -1, :].unsqueeze(1)
                p_next = self.planner.step_plan(act_emb, p_curr)
                
                # Apply the planner update ONLY to the sequences that actually generated a comma
                mask_expanded = is_comma.view(b_size, 1, 1).expand_as(p_curr)
                p_curr = torch.where(mask_expanded, p_next, p_curr)

            if not unfinished_sequences.any():
                break
                
            # Prepare loop tensors for the single new token
            current_input_ids = next_token
            current_plan = p_curr
            current_film_mask = torch.ones((b_size, 1), device=device)
            curr_pos_ids = curr_pos_ids[:, -1:] + 1
            
            # Extend attention mask
            new_attn_mask = torch.ones((b_size, 1), dtype=torch.long, device=device)
            attention_mask = torch.cat([attention_mask, new_attn_mask], dim=1)

        return generated_ids
    
    def save_pretrained(self, save_path):
        """
        Saves the entire LatentWriter model and tokenizer to the specified path.
        """
        torch.save(self.state_dict(), f"{save_path}/writer_model.pt")