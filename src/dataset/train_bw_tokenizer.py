"""Train a custom tokenizer for the Blocksworld dataset using the tokenizers library."""

import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, Regex

def train_blocksworld_tokenizer(data_path="./data_cache/blocksworld_lexical/train.jsonl"):
    """Train a custom tokenizer for the Blocksworld dataset."""
    save_path="./src/config/tokenizer"
    os.makedirs(save_path, exist_ok=True)
    print("Training custom Blocksworld tokenizer...")

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r","), behavior="isolated"),
        pre_tokenizers.WhitespaceSplit()
    ])
    special_tokens = ["[UNK]", "[PAD]", "[INIT]", "[GOAL]", "[STEPS]", "[DONE]", ","]

    trainer = trainers.WordLevelTrainer(
        vocab_size=100,
        special_tokens=special_tokens
    )

    def get_training_corpus():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    yield sample["prompt"]
                    yield sample["trajectory"]

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    file_path = os.path.join(save_path, "bw6_tokenizer.json")
    tokenizer.save(file_path)

    print(f"Tokenizer trained and saved to {file_path}")
    print(f"Final Vocabulary Size: {tokenizer.get_vocab_size()}")

    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    print("\nLearned Vocabulary:")
    print([v[0] for v in sorted_vocab])

if __name__ == "__main__":
    train_blocksworld_tokenizer("./data_cache/blocksworld_lexical/train.jsonl")
