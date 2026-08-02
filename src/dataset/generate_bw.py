"""Generate synthetic blocksworld datasets for training and evaluation."""

import json
import random
from collections import deque

# 1. Define the Blocks
BLOCKS = ["A", "B", "C", "D"]


def is_goal_met(current_state, goal_state):
    """Check if all conditions in the goal state exist in the current state."""
    return all(cond in current_state for cond in goal_state)


def get_valid_actions(state, block_names=None):
    """Return valid action/state pairs based on PDDL rules."""
    if block_names is None:
        block_names = BLOCKS

    actions = []

    # Extract properties
    clear = [block for block in block_names if f"clear {block}" in state]
    on_table = [block for block in block_names if f"{block} on table" in state]
    on = {
        block: other
        for block in block_names
        for other in block_names
        if f"{block} on {other}" in state
    }
    holding = [block for block in block_names if f"holding {block}" in state]

    # Rule 1: PICKUP (from table)
    if not holding:
        for block in clear:
            if block in on_table:
                new_state = set(state) - {f"{block} on table", f"clear {block}", "hand empty"}
                new_state.update({f"holding {block}"})
                actions.append((f"pickup {block}", new_state))

    # Rule 2: UNSTACK (from another block)
    if not holding:
        for block in clear:
            if block in on:
                under_block = on[block]
                new_state = set(state) - {
                    f"{block} on {under_block}",
                    f"clear {block}",
                    "hand empty",
                }
                new_state.update({f"holding {block}", f"clear {under_block}"})
                actions.append((f"unstack {block} {under_block}", new_state))

    # Rule 3: PUTDOWN (to table)
    if holding:
        block = holding[0]
        new_state = set(state) - {f"holding {block}"}
        new_state.update({f"{block} on table", f"clear {block}", "hand empty"})
        actions.append((f"putdown {block}", new_state))

    # Rule 4: STACK (onto another block)
    if holding:
        block = holding[0]
        for other in clear:
            new_state = set(state) - {f"holding {block}", f"clear {other}"}
            new_state.update({f"{block} on {other}", f"clear {block}", "hand empty"})
            actions.append((f"stack {block} {other}", new_state))

    return actions


def bfs_solve(init_state, goal_state, block_names=None):
    """Find the shortest sequence of actions from init to goal."""
    # Queue stores tuples of (current_state, path_of_actions)
    queue = deque([(set(init_state), [])])
    visited = set([frozenset(init_state)])

    while queue:
        current_state, path = queue.popleft()

        if is_goal_met(current_state, goal_state):
            return path

        for action_str, new_state in get_valid_actions(current_state, block_names):
            frozen_new = frozenset(new_state)
            if frozen_new not in visited:
                visited.add(frozen_new)
                queue.append((new_state, path + [action_str]))

    return None

# =====================================================================
# 2. PROCEDURAL STATE GENERATION (The Fix)
# =====================================================================

def get_base_state(num_blocks=5, block_names=None):
    """Return a state where all blocks are clear and on the table."""
    if block_names is None:
        block_names = BLOCKS[:num_blocks]
    state = {"hand empty"}
    for block in block_names[:num_blocks]:
        state.add(f"{block} on table")
        state.add(f"clear {block}")
    return frozenset(state)


def generate_random_state(num_blocks=5, num_shuffles=20, block_names=None):
    """Generate a random valid state by taking a random walk from the base state."""
    current_state = get_base_state(num_blocks, block_names)

    for _ in range(num_shuffles):
        valid_actions = get_valid_actions(current_state, block_names)
        if not valid_actions:
            break
        # Pick a random valid action and apply it
        _, next_state = random.choice(valid_actions)
        current_state = frozenset(next_state)

    return current_state

def generate_dataset(num_samples=5000, num_blocks=5, min_steps=4,
                     save_path="blocksworld_train.jsonl"):
    """Procedurally generate unique Blocksworld problems."""
    block_names = [chr(ord("A") + index) for index in range(num_blocks)]

    dataset = []
    seen_configs = set()

    print(f"Generating {num_samples} procedural Blocksworld samples...")

    while len(dataset) < num_samples:
        # 1. Generate two random states
        init_state = generate_random_state(num_blocks, block_names=block_names)
        goal_state = generate_random_state(num_blocks, block_names=block_names)

        # 2. Ensure they are distinct and not already in the dataset
        config_hash = (init_state, goal_state)
        if init_state == goal_state or config_hash in seen_configs:
            continue

        # 3. Filter the goal to only contain useful block relationships.
        relaxed_goal = {cond for cond in goal_state if "on" in cond}

        # 4. Solve it!
        optimal_path = bfs_solve(init_state, relaxed_goal, block_names)

        # 5. Only keep problems that require actual planning.
        if optimal_path and len(optimal_path) >= min_steps:
            seen_configs.add(config_hash)
            dataset.append(
                {
                    "prompt": (
                        f"[INIT] {', '.join(sorted(init_state))} "
                        f"[GOAL] {', '.join(sorted(relaxed_goal))}"
                    ),
                    "trajectory": ", ".join(optimal_path) + " [DONE]",
                    "steps": len(optimal_path),
                }
            )

            if len(dataset) % 1000 == 0:
                print(f"  Generated {len(dataset)} / {num_samples}...")

    # Save to JSONL
    with open(save_path, "w", encoding="utf-8") as file_handle:
        for sample in dataset:
            file_handle.write(json.dumps(sample) + "\n")

    print(f"Success! Saved {len(dataset)} unique procedural Blocksworld samples.")


if __name__ == "__main__":
    generate_dataset(
        num_samples=10000,
        num_blocks=5,
        min_steps=4,
        save_path="./data_cache/blocksworld/blocksworld_train.jsonl",
    )
    generate_dataset(
        num_samples=1000,
        num_blocks=6,
        min_steps=5,
        save_path="./data_cache/blocksworld/blocksworld_eval_6blocks.jsonl",
    )
    generate_dataset(
        num_samples=1000,
        num_blocks=4,
        min_steps=5,
        save_path="./data_cache/blocksworld/blocksworld_eval_4blocks.jsonl",
    )
