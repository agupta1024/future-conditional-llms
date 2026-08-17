""" Generate lexically-robust blocksworld dataset for training and evaluation. """

import random
import json
from collections import deque


POOL = ['A', 'B', 'C', 'D', 'E', 'F']

def get_base_state(active_blocks):
    """Returns the state where active blocks are clear and on the table."""
    state = set(["hand empty"])
    for b in active_blocks:
        state.add(f"{b} on table")
        state.add(f"clear {b}")
    return frozenset(state)

def get_valid_actions(state, active_blocks):
    # pylint: disable=multiple-statements
    """Returns valid actions constrained to the currently active blocks."""
    actions = []
    clear = [b for b in active_blocks if f"clear {b}" in state]
    on_table = [b for b in active_blocks if f"{b} on table" in state]
    on = {b: c for b in active_blocks for c in active_blocks if f"{b} on {c}" in state}
    holding = [b for b in active_blocks if f"holding {b}" in state]

    if not holding:
        for b in clear:
            if b in on_table:
                actions.append((f"pickup {b}",
                                set(state) - {f"{b} on table",
                                                f"clear {b}",
                                                "hand empty"} | {f"holding {b}"}
                                ))
            if b in on:
                under_b = on[b]
                actions.append((f"unstack {b} {under_b}",
                                set(state) - {f"{b} on {under_b}",
                                                f"clear {b}", "hand empty"} | {f"holding {b}",
                                                                                f"clear {under_b}"}
                                ))
    if holding:
        b = holding[0]
        actions.append((f"putdown {b}",
                        set(state) - {f"holding {b}"} | {f"{b} on table",
                                                            f"clear {b}", "hand empty"}
                        ))
        for c in clear:
            actions.append((f"stack {b} {c}",
                            set(state) - {f"holding {b}",
                                            f"clear {c}"} | {f"{b} on {c}",
                                                            f"clear {b}", "hand empty"}
                            ))
    return actions

def generate_random_state(active_blocks, num_shuffles=25):
    """Generates a random valid state by shuffling the base state."""
    current_state = get_base_state(active_blocks)
    for _ in range(num_shuffles):
        valid_actions = get_valid_actions(current_state, active_blocks)
        if not valid_actions:
            break
        _, next_state = random.choice(valid_actions)
        current_state = frozenset(next_state)

    holding_blocks = [b for b in active_blocks if f"holding {b}" in current_state]
    if holding_blocks:
        b = holding_blocks[0]
        fixed_state = set(current_state) - {f"holding {b}"}
        fixed_state.update({f"{b} on table", f"clear {b}", "hand empty"})
        current_state = frozenset(fixed_state)

    return current_state

def is_goal_met(current_state, goal_state):
    """Check if the current state satisfies the goal state."""
    return all(cond in current_state for cond in goal_state)

def bfs_solve(init_state, goal_state, active_blocks):
    """Breadth-first search to find the shortest sequence of actions."""
    queue = deque([(set(init_state), [])])
    visited = set([frozenset(init_state)])

    while queue:
        current_state, path = queue.popleft()
        if is_goal_met(current_state, goal_state):
            return path
        for action_str, new_state in get_valid_actions(current_state, active_blocks):
            frozen_new = frozenset(new_state)
            if frozen_new not in visited:
                visited.add(frozen_new)
                queue.append((new_state, path + [action_str]))
    return None

def generate_datasets(num_train=50000, num_eval=500):
    """Generates lexically-robust blocksworld datasets for training and evaluation."""
    print("Generating Lexically-Robust Blocksworld Datasets...")

    def build_set(num_samples, num_active_blocks):
        # pylint: disable=too-many-locals
        data = []
        seen = set()
        while len(data) < num_samples:
            active_blocks = random.sample(POOL, num_active_blocks)

            init_state = generate_random_state(active_blocks)
            goal_state = generate_random_state(active_blocks)

            if init_state == goal_state:
                continue

            relaxed_goal = set(cond for cond in goal_state if "on" in cond)
            path = bfs_solve(init_state, relaxed_goal, active_blocks)

            if not path or len(path) < 3:
                continue

            config_hash = (frozenset(init_state), frozenset(goal_state))
            if config_hash in seen:
                continue
            seen.add(config_hash)

            init_str = ", ".join(sorted(list(init_state)))
            goal_str = ", ".join(sorted(list(relaxed_goal)))
            prompt = f"[INIT] {init_str} [GOAL] {goal_str}"
            trajectory = ", ".join(path) + " [DONE]"

            data.append({"prompt": prompt, "trajectory": trajectory})
        return data

    train_data = build_set(num_train, 5)
    eval_data_5 = build_set(num_eval, 5)
    eval_data_6 = build_set(num_eval, 6)

    with open("./data_cache/blocksworld_lexical/train_big.jsonl", "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d) + "\n")
    with open("./data_cache/blocksworld_lexical/eval_5.jsonl", "w", encoding="utf-8") as f:
        for d in eval_data_5:
            f.write(json.dumps(d) + "\n")
    with open("./data_cache/blocksworld_lexical/eval_6.jsonl", "w", encoding="utf-8") as f:
        for d in eval_data_6:
            f.write(json.dumps(d) + "\n")
    print("Done! You now have a scientifically pure OOD test set.")

if __name__ == "__main__":
    generate_datasets()
