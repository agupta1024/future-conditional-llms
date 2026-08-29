# Future-Conditional LLMs

Research code for conditioning an autoregressive language model on a learned
representation of future text or future actions. The project compares a
standard GPT-2-style baseline with a two-stage model consisting of a latent
predictor and a latent-conditioned writer.

## Project Layout

```text
src/config/                     Dataset and model/checkpoint configuration
src/dataset/                    Dataset generation, tokenisation, and loaders
src/model/                      Baseline, predictor, and writer implementations
src/evaluation/                 Task evaluation and plotting/benchmark scripts
data_cache/                     Generated JSONL data (not included by default)
<dataset>_<stage>_model_<size>/ Saved checkpoints and final model weights
```

The supported task families are:

* **Treasure Hunt**: long-context retrieval and generation with a target and
	passkey embedded in the context.
* **Blocksworld**: action-sequence generation evaluated by a symbolic
	simulator.

## Setup

Use Python 3.10+ and a CUDA-enabled PyTorch installation for practical
training runs. Install the repository dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from the repository root. The source files use relative package
imports, so `python -m ...` is required for the modules below.

## Data Preparation

Dataset configuration lives in `src/config/dataset_config.py`. The expected
paths are:

```text
data_cache/treasure_hunt/train.jsonl
data_cache/treasure_hunt/eval_horizon_64.jsonl
data_cache/blocksworld/train.jsonl
data_cache/blocksworld/eval.jsonl
```

Generate task data with the corresponding modules when a fresh dataset is
needed:

```bash
python -m src.dataset.generate_th
python -m src.dataset.generate_bw_sub
python -m src.dataset.prepare_dataset_th
python -m src.dataset.prepare_dataset_bw
python -m src.dataset.prepare_ds_goal_drop
```

The generator modules use their settings in the source file rather than
command-line arguments. For Blocksworld, train the WordLevel tokenizer
after generating the JSONL data:

```bash
python -m src.dataset.train_bw_tokenizer
```

This writes `src/config/tokenizer/bw6_tokenizer.json`, which is the tokenizer
configured for the Blocksworld variants.

## Model Stages

Training is organised as follows:

1. **Base**: train a GPT-2-style autoregressive model from scratch.
2. **Predictor**: train the future representation/predictor with VICReg loss
	 while using the base model as its encoder.
3. **Writer**: train a writer conditioned on the predictor representation.

The model sizes are configured in `src/config/model_config.py`:

| Name | Context | Layers | Hidden size | Use |
| --- | ---: | ---: | ---: | --- |
| `gpt2_512` | 512 | 4 | 512 | Treasure Hunt writer 85M|
| `gpt2_512-l` | 512 | 20 | 512 | Baseline 89M (Parameter-matched) |
| `gpt2_1024` | 1024 | 4 | 512 | Blocksworld writer 32M|
| `gpt2_1024-s` | 1024 | 10 | 512 | Blocksworld baseline 32M  (Parameter-matched)|

## Training

The training modules currently select the dataset, model size, epochs, and
output directory in their source-level `__main__` calls. Edit those values
before running a different dataset or variant.

### Base model

`train_gpt2.py` trains the scratch autoregressive baseline and saves
`base_model.pt`, plus periodic checkpoints:

```bash
python -m src.model.train_gpt2
```

The default is Blocksworld with `gpt2_1024-s`; change the final
`train(dataset_name=...)` call for `treasure_hunt`.

### Latent predictor

`train_planner.py` trains the future predictor with VICReg loss and saves
`planner_model.pt`:

```bash
python -m src.model.train_planner
```

The default is Treasure Hunt with `gpt2_512` encoder. The function accepts
`treasure_hunt`, or `blocksworld` as `dataset_name`.

### Writer model

For Treasure Hunt, use the DDP-enabled writer trainer. It
expects multiple CUDA processes and defaults to Treasure Hunt:

```bash
torchrun --standalone --nproc_per_node=2 -m src.model.train_th_writer
```

For Blocksworld, use the end-to-end continuous-plan trainer instead:

```bash
python -m src.model.train_bw_writer
```

The Blocksworld trainer accepts an optional `objective="_goal_drop"` in its
source-level call and writes to a matching `_goal_drop` directory. The
alternative `train_bw_static_writer.py` trains a static plan variant:

```bash
python -m src.model.train_bw_static_writer
```

Typical output directories include
`blocksworld_base_model_gpt2_1024-s`,
`blocksworld_planner_model_gpt2_1024`, and
`blocksworld_writer_model_gpt2_1024`. The loader resolves these names from the
selected dataset, stage, and model size; custom paths can be passed through
`get_model_and_tokenizer` when composing a different experiment.

## Evaluation and Benchmarks

Blocksworld evaluation loads the configured writer and baseline checkpoints,
simulates generated action sequences, and writes
`bw_benchmarks/eval_results.json`:

```bash
python -m src.evaluation.bw_eval_model
```

It reports perfect-plan success, partial completion, and
inference efficiency for different seeds and context windows 8, 16, 32, and 64.

Treasure Hunt evaluation compares retrieval and efficiency over several
horizons, context windows, and seeds. It writes one result file per seed to
`th_benchmarks/` and then runs the aggregate analysis:

```bash
python -m src.evaluation.eval_th
```

Additional evaluation utilities include:

* `src.evaluation.bw_ablation`: Blocksworld ablation experiments.
* `src.evaluation.bw_resilience_eval`: Blocksworld resilience tests.
* `src.evaluation.latent_control_benchmark`: latent-control benchmark.
* `src.evaluation.plot_metrics`: efficiency profiling and metric plots.

These utilities also use source-level defaults for model paths and dataset
locations. Inspect the module before running an experiment with a custom
checkpoint.

## Configuration and Reproducibility

Edit `src/config/dataset_config.py` to change data paths, tokenizer paths,
vocabulary sizes, or batch sizes. Edit `src/config/model_config.py` to add or
modify model dimensions and checkpoint resolution. Several training scripts 
set seed `42`; Evaluations additionally run seeds `1337`, `2024`, `7777`, and `9999`.

Weights are loaded with `torch.load` and may be raw state dictionaries or
checkpoint dictionaries containing `model_state_dict`. Ensure the directory
names match the configured dataset/stage/model combination before evaluation.

## Citation

This repository contains research code for future-conditional language model
experiments. Add the project citation or paper reference here when the
associated publication details are finalized.
