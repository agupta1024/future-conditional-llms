"""Dataset configuration helpers for the training and evaluation pipelines."""


def get_dataset_config(name="tinystories", max_seq_length=1024, **_kwargs):
    """
    Returns a dictionary containing the configuration for the dataset.
    This function can be extended to include more parameters as needed.
    """
    if name == "tinystories":
        dataset_config = {
            "name": "tinystories",
            "split": "train",
            "max_seq_length": 512,
            "batch_size": 32,
            "dataset_size": 100000,
            "train_path": "./data_cache/",
            "val_path": "./data_cache/",
        }
    elif name == "treasure_hunt":
        dataset_config = {
            "name": "treasure_hunt",
            "split": "train",
            "max_seq_length": 512,
            "batch_size": 32,
            "dataset_size": 100000,
            "train_path": "./data_cache/treasure_hunt/train.jsonl",
            "val_path": "./data_cache/treasure_hunt/eval_horizon_64.jsonl",
            "val_128_path": "./data_cache/treasure_hunt/eval_horizon_128.jsonl",
            "val_256_path": "./data_cache/treasure_hunt/eval_horizon_256.jsonl",
            "val_512_path": "./data_cache/treasure_hunt/eval_horizon_512.jsonl",
        }
    elif name == "blocksworld":
        dataset_config = {
            "name": "blocksworld",
            "split": "train",
            "max_seq_length": 512,
            "batch_size": 32,
            "dataset_size": 10000,
            "train_path": "./data_cache/blocksworld/blocksworld_train.jsonl",
            "val_path": "./data_cache/blocksworld/blocksworld_eval.jsonl",
            "val_4blocks_path": "./data_cache/blocksworld/blocksworld_eval_4blocks.jsonl",
            "val_6blocks_path": "./data_cache/blocksworld/blocksworld_eval_6blocks.jsonl",
        }
    elif name == "blocksworld_lexical":
        dataset_config = {
            "name": "blocksworld_lexical",
            "split": "train",
            "max_seq_length": 512,
            "batch_size": 32,
            "dataset_size": 10000,
            "train_path": "./data_cache/blocksworld_lexical/train.jsonl",
            "train_big_path": "./data_cache/blocksworld_lexical/train_big.jsonl",
            "val_path": "./data_cache/blocksworld_lexical/eval.jsonl",
            "val6_path": "./data_cache/blocksworld_lexical/eval_6.jsonl",
        }
    elif name == "blocksworld_sub":
        dataset_config = {
            "name": "blocksworld_sub",
            "split": "train",
            "max_seq_length": 512,
            "batch_size": 32,
            "dataset_size": 10000,
            "train_path": "./data_cache/blocksworld_sub/train.jsonl",
            "train_big_path": "./data_cache/blocksworld_sub/train_big.jsonl",
            "val_path": "./data_cache/blocksworld_sub/eval.jsonl",
        }
    else:
        path = "./data_cache"
        dataset_config = {
            "name": name,
            "split": "train",
            "max_seq_length": max_seq_length,
            "batch_size": 16,
            "dataset_size": 100000,
            "train_path": f"{path}/train",
            "val_path": f"{path}/validation",
        }
    return dataset_config
