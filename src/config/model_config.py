"""Model and tokenizer configuration helpers for training and evaluation."""

import os

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

from ..model.ar_gpt2 import GPT2Baseline
from ..model.model_ts import LatentPlanner as PlannerTS
from ..model.model_ts import LatentWriter as WriterTS

from ..model.model_bw import LatentPlanner as PlannerBW
from ..model.model_bw import LatentWriter as WriterBW

from ..model.model_th import LatentPlanner as PlannerTH
from ..model.model_th import LatentWriter as WriterTH

_MODEL_DIM_CONFIG = {
    "512": {
        "n_embd": 512,
        "n_layer": 4,
        "n_head": 8,
        "z_latent": 2048,
    },
    "1024": {
        "n_embd": 512,
        "n_layer": 4,
        "n_head": 8,
        "z_latent": 512,
    },
    "512-l": {
        "n_embd": 512,
        "n_layer": 20,
        "n_head": 8,
        "z_latent": 2048,
    },
    "1024-s": {
        "n_embd": 512,
        "n_layer": 10,
        "n_head": 8,
        "z_latent": 512,
    },
    "1024-l": {
        "n_embd": 512,
        "n_layer": 20,
        "n_head": 8,
        "z_latent": 512,
    }
}


def _build_tokenizer(tokenizer_path=None):
    """Create the shared GPT-2 tokenizer used across model variants."""
    if tokenizer_path:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'eos_token': '[PAD]',
            'additional_special_tokens': ['[INIT]', '[GOAL]', ',']
        })
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_gpt2_config(tokenizer, model_config_key, max_seq_length, vocab_size=None):
    """Build a GPT-2 config without relying on keyword-only constructor stubs."""
    dim_config = _MODEL_DIM_CONFIG[model_config_key]
    gpt2_config = GPT2Config()
    gpt2_config.vocab_size = vocab_size if vocab_size else len(tokenizer)
    gpt2_config.n_positions = max_seq_length
    gpt2_config.n_embd = dim_config["n_embd"]
    gpt2_config.n_layer = dim_config["n_layer"]
    gpt2_config.n_head = dim_config["n_head"]
    gpt2_config.bos_token_id = tokenizer.bos_token_id
    gpt2_config.eos_token_id = tokenizer.eos_token_id
    gpt2_config.pad_token_id = tokenizer.pad_token_id
    return gpt2_config


def _resolve_model_paths(
    dataset_name,
    working_model,
    custom_ar,
    base_model_path,
    writer_model_path,
    planner_model_path,
): # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Resolve the checkpoint paths for the selected model family."""
    if base_model_path is None:
        base_model_path = f"./{dataset_name}_base_model_{working_model}"
        if custom_ar:
            base_model_path = base_model_path + "/base_model.pt"

    planner_path = f"./{dataset_name}_planner_model_{working_model}/planner_model.pt"
    if planner_model_path is not None:
        planner_path = planner_model_path

    writer_path = f"./{dataset_name}_writer_model_{working_model}/writer_model.pt"
    if writer_model_path is not None:
        writer_path = writer_model_path

    return base_model_path, planner_path, writer_path


def _resize_position_embeddings(model, max_seq_length):
    """Resize GPT-2 position embeddings when the requested context is larger."""
    if model.transformer.wpe.weight.shape[0] >= max_seq_length:
        return model

    old_wpe = model.transformer.wpe.weight.data
    new_wpe = torch.nn.Embedding(max_seq_length, model.config.n_embd)
    with torch.no_grad():
        new_wpe.weight.data[: old_wpe.shape[0]] = old_wpe
    model.transformer.wpe = new_wpe
    model.config.n_positions = max_seq_length
    model.config.max_position_embeddings = max_seq_length
    print(f"Positional embeddings resized to: {model.transformer.wpe.weight.shape}")
    return model

def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """Load a model checkpoint from the specified path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint file found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    loaded_data = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded_data, dict) and "model_state_dict" in loaded_data:
        model_state = loaded_data["model_state_dict"]
        start_step = loaded_data.get("step", 0)

        if optimizer is not None and "optimizer_state_dict" in loaded_data:
            optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
            print("Successfully loaded optimizer state.")
    else:
        model_state = loaded_data
        start_step = 0
        print("Raw model state_dict detected (no optimizer state).")

    cleaned_state_dict = {key.replace("module.", ""): value for key, value in model_state.items()}
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Model loaded successfully! Resuming at Step: {start_step}")

    return model, optimizer, start_step


def _create_base_model(
    config,
    base_model_path,
    custom_ar,
    load_scratch,
    fetch_from_hf,
    max_seq_length,
    device,
): # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Create the base model variant for the requested checkpoint source."""
    if load_scratch:
        if custom_ar:
            print("Initializing custom baseline model from scratch...")
            return GPT2Baseline(config)
        return GPT2LMHeadModel(config)

    if fetch_from_hf:
        base_model_path = "gpt2"

    if custom_ar:
        print(f"Loading custom baseline model from {base_model_path}...")
        model = GPT2Baseline(config)
        model, _, _ = load_checkpoint(model, base_model_path, device=device)
        return model

    print(f"Fetching pretrained model from Hugging Face: {base_model_path}...")
    model = GPT2LMHeadModel.from_pretrained(base_model_path)
    return _resize_position_embeddings(model, max_seq_length)


def _create_planner_model(
    config,
    dataset_name,
    base_model_path,
    custom_ar,
    device,
    model_config_key,
): # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Create the planner model for the selected dataset family."""
    print(f"Loading base model from {base_model_path}...")
    if dataset_name == "tinystories":
        planner_cls = PlannerTS
    elif dataset_name == "treasure_hunt":
        planner_cls = PlannerTH
    else:
        planner_cls = PlannerBW

    return planner_cls(
        model_name_or_path=base_model_path,
        latent_dim=_MODEL_DIM_CONFIG[model_config_key]["z_latent"],
        custom_ar=custom_ar,
        config=config,
    ).to(device)


def _create_writer_model(
    config,
    dataset_name,
    base_model_path,
    custom_ar,
    film,
    device,
    model_config_key,
    max_seq_length,
): # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Create the writer model for the selected dataset family."""
    if dataset_name == "tinystories":
        writer_cls = WriterTS
    elif dataset_name == "treasure_hunt":
        writer_cls = WriterTH
    else:
        writer_cls = WriterBW

    writer_kwargs = {
        "base_model_path": base_model_path,
        "vocab_size": config.vocab_size,
        "max_seq_len": max_seq_length,
        "latent_dim": _MODEL_DIM_CONFIG[model_config_key]["z_latent"],
        "custom_ar": custom_ar,
        "config": config,
    }
    if writer_cls is WriterTS:
        writer_kwargs["film"] = film
    if writer_cls is WriterBW:
        writer_kwargs["latent_dim"] = 512
    return writer_cls(**writer_kwargs).to(device)


def get_model_and_tokenizer(
    working_model="gpt2",
    max_seq_length=1024,
    load_scratch=False,
    dataset_name="tinystories",
    custom_ar=False,
    load_stage="base",
    fetch_from_hf=False,
    film=False,
    base_model_path=None,
    writer_model_path=None,
    planner_model_path=None,
    overwrite_base=False,
    overwrite_planner=False,
    vocab_size=None,
    tokenizer_path=None,
):
    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    """Return a tokenizer and the requested model variant."""
    tokenizer = _build_tokenizer(tokenizer_path=tokenizer_path)
    model_config_key = working_model.split('_')[1]
    gpt2_config = _build_gpt2_config(tokenizer, model_config_key, max_seq_length,
                                         vocab_size)
    print(f"Initializing fresh {dataset_name} {load_stage} for {working_model}...")

    base_model_path, planner_path, writer_path = _resolve_model_paths(
        dataset_name,
        working_model,
        custom_ar,
        base_model_path,
        writer_model_path,
        planner_model_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_stage == "base":
        model = _create_base_model(
            gpt2_config,
            base_model_path,
            custom_ar,
            load_scratch,
            fetch_from_hf,
            max_seq_length,
            device,
        )
    elif load_stage == "planner":
        model = _create_planner_model(
            gpt2_config,
            dataset_name,
            base_model_path,
            custom_ar,
            device,
            model_config_key,
        )
        if not load_scratch:
            model, _, _ = load_checkpoint(model, planner_path, device=device)
    else:
        model = _create_writer_model(
            gpt2_config,
            dataset_name,
            base_model_path,
            custom_ar,
            film,
            device,
            model_config_key,
            max_seq_length
        )
        if not load_scratch:
            model, _, _ = load_checkpoint(model, writer_path, device=device)
        if overwrite_planner:
            print(f"Overwriting planner model weights from {planner_path}...")
            model.planner, _, _ = load_checkpoint(
                model.planner,
                planner_path,
                device=device,
            )
        if overwrite_base:
            print(f"Overwriting base model weights from {base_model_path}...")
            model.planner.encoder, _, _ = load_checkpoint(
                model.planner.encoder,
                base_model_path,
                device=device,
            )

    return tokenizer, model
