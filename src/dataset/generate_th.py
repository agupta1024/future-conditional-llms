"""Generate synthetic treasure-hunt datasets for training and evaluation."""

import json
import os
import random

# We define distinct lists to ensure the model has to learn to map the specific
# prompt variables to the final action, rather than memorizing a single story.
ITEMS = [
    "Ruby Amulet",
    "Sapphire Ring",
    "Iron Key",
    "Golden Idol",
    "Silver Coin",
    "Crystal Skull",
    "Ancient Scroll",
    "Brass Compass",
    "Emerald Dagger",
    "Platinum Watch",
    "Jade Pendant",
    "Onyx Statue",
]

ROOMS = [
    "Basement",
    "Attic",
    "Observatory",
    "Library",
    "Wine Cellar",
    "Greenhouse",
    "Master Bedroom",
    "Servants Quarters",
    "Kitchen",
    "Dining Hall",
    "Armory",
    "Trophy Room",
]

FILLER_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)

# Word mapping for the 'words' format
DIGIT_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

# Distractor sentences to act as the "memory buffer" (Trajectory)
# These simulate the agent taking steps that push the prompt out of the local context window.
DISTRACTORS = [
    "I walked down the long, dusty hallway.",
    "I opened a wooden door, but it was just an empty broom closet.",
    "A mouse scurried across the floorboards, startling me.",
    "I checked the drawers of an old desk, finding only moth-eaten papers.",
    "The moonlight shone through the cracked window.",
    "I thought I heard footsteps upstairs, so I paused to listen.",
    "There was a painting of a stern-looking man on the wall.",
    "I tripped over a loose rug and almost dropped my flashlight.",
    "The air grew colder as I moved deeper into the mansion.",
    "I tried the handle of the guest room, but it was firmly locked.",
    "Cobwebs brushed against my face as I ducked under a low beam.",
    "I took a moment to catch my breath and check my surroundings.",
    "A grandfather clock chimed loudly in the distance.",
    "I noticed a strange set of scratch marks on the wallpaper.",
    "The floorboards creaked loudly under my weight.",
    "I peered out the window, but the grounds were pitch black.",
    "A sudden gust of wind blew out the candle I was holding.",
    "I found a pile of old books stacked haphazardly in the corner.",
    "There was a faint smell of damp earth in the air.",
    "I wiped the dust off a mirror, revealing my own tired reflection."
]

def generate_mohtashami_haystack(target_tokens, tokenizer):
    """Generate the exact background noise used in the Passkey benchmark."""
    haystack = ""
    while len(tokenizer.encode(haystack)) < target_tokens:
        haystack += FILLER_SENTENCE
    return haystack


def format_passkey(code_str, mode):
    """Format the 4-digit code into the requested ablation mode."""
    if mode == "raw":
        return code_str
    if mode == "hyphen":
        return "-".join(list(code_str))
    return " ".join(DIGIT_TO_WORD[digit] for digit in code_str)

def generate_filler(target_word_count):
    """
    Generates a string of random distractor sentences to artificially inflate
    the sequence length, pushing the model's memory to its limits.
    """
    filler_sentences = []
    current_word_count = 0
    deck = list(DISTRACTORS)
    random.shuffle(deck)
    while current_word_count < target_word_count:
        if not deck:
            deck = list(DISTRACTORS)
            random.shuffle(deck)
        if filler_sentences and deck[0] == filler_sentences[-1]:
            deck.append(deck.pop(0))

        sentence = deck.pop(0)
        filler_sentences.append(sentence)
        current_word_count += len(sentence.split())

    return filler_sentences

def generate_sample(target_length, mode="words"):
    """
    Creates a single, deterministic Treasure Hunt log.
    """
    # 1. Pick the factual variables
    item = random.choice(ITEMS)
    room = random.choice(ROOMS)
    code = f"{random.randint(1000, 9999)}"
    _ = format_passkey(code, mode)
    # 2. Formulate the Prompt (This goes into your Planner/FiLM layer)
    prompt = f"[GOAL] Target: {item} | Code: {code}\n[TRAJECTORY] "

    filler_sentences = generate_filler(target_length)
    clue = f"I checked my map and realized the vault was hidden in the {room}."
    filler_sentences.append(clue)
    # injection_zone = min(5, len(filler_sentences))
    # inject_idx = len(filler_sentences) - random.randint(1, injection_zone)
    # filler_sentences.insert(inject_idx, clue)

    filler_text = " ".join(filler_sentences)
    final_action = (
        f" I finally arrived at the {room}. I approached the lockpad and "
        f"entered {code}. The vault clicked open, and I retrieved the {item}. "
        "<|endoftext|>"
    )

    full_text = prompt + filler_text + final_action

    return {
        "prompt": prompt,
        "filler": filler_text,
        "final_action": final_action,
        "full_text": full_text,
        "metadata": {
            "item": item,
            "room": room,
            "code": code,
            "filler_word_count": len(filler_text.split()),
        }
    }

def build_dataset(num_samples, filename, min_len=50, max_len=300):
    """Generate a JSONL file containing generated samples."""
    print(f"Generating {num_samples} samples for {filename}...")
    dataset = []

    try:
        with open(filename, "w", encoding="utf-8") as file_handle:
            for _ in range(num_samples):
                target_length = random.randint(min_len, max_len)
                sample = generate_sample(target_length)

                file_handle.write(json.dumps(sample) + "\n")
                dataset.append(sample)

        print(f"Successfully saved {filename}")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Error saving dataset: {exc}")

    return dataset

if __name__ == "__main__":
    os.makedirs("./data_cache/treasure_hunt", exist_ok=True)

    build_dataset(
        num_samples=10000,
        filename="./data_cache/treasure_hunt/train.jsonl",
        min_len=50,
        max_len=300,
    )
    eval_horizons = [64, 128, 256, 512]

    for horizon in eval_horizons:
        build_dataset(
            num_samples=500,
            filename=f"./data_cache/treasure_hunt/eval_horizon_{horizon}.jsonl",
            min_len=horizon,
            max_len=horizon + 10,
        )

    print("\nDataset generation complete! You are ready to train.")
