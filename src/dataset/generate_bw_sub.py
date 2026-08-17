""" Generate synthetic blocksworld dataset for training and evaluation. """

import random
import json
from collections import deque

BLOCKS = ['A', 'B', 'C', 'D', 'E']

def is_goal_met(current_state, goal_state):
    """Check if the current state satisfies the goal state."""
    return all(cond in current_state for cond in goal_state)

def get_valid_actions(state):
    # pylint: disable=multiple-statements
    """Returns valid actions constrained to the currently active blocks."""
    actions = []
    clear = [b for b in BLOCKS if f"clear {b}" in state]
    on_table = [b for b in BLOCKS if f"{b} on table" in state]
    on = {b: c for b in BLOCKS for c in BLOCKS if f"{b} on {c}" in state}
    holding = [b for b in BLOCKS if f"holding {b}" in state]

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

def get_inverse_action(action_str):
    """Returns the exact logical inverse to undo a move perfectly."""
    parts = action_str.split()
    cmd = parts[0]
    if cmd == "pickup":
        return f"putdown {parts[1]}"
    if cmd == "putdown":
        return f"pickup {parts[1]}"
    if cmd == "unstack":
        return f"stack {parts[1]} {parts[2]}"
    if cmd == "stack":
        return f"unstack {parts[1]} {parts[2]}"
    return None

def bfs_solve(init_state, goal_state):
    """Breadth-first search to find the shortest sequence of actions."""
    queue = deque([(set(init_state), [])])
    visited = set([frozenset(init_state)])

    while queue:
        current_state, path = queue.popleft()
        if is_goal_met(current_state, goal_state):
            return path
        for action_str, new_state in get_valid_actions(current_state):
            frozen_new = frozenset(new_state)
            if frozen_new not in visited:
                visited.add(frozen_new)
                queue.append((new_state, path + [action_str]))
    return None

def get_base_state():
    """Returns the base state where all blocks are on the table and clear, and the hand is empty."""
    state = set(["hand empty"])
    for b in BLOCKS:
        state.add(f"{b} on table")
        state.add(f"clear {b}")
    return frozenset(state)

def generate_random_state(num_shuffles=25):
    """Generates a random valid state by performing random valid actions from the base state."""
    current_state = get_base_state()
    for _ in range(num_shuffles):
        valid_actions = get_valid_actions(current_state)
        if not valid_actions:
            break
        _, next_state = random.choice(valid_actions)
        current_state = frozenset(next_state)

    # Force hand empty so we don't need the word [UNK] "holding" in the prompt!
    holding_blocks = [b for b in BLOCKS if f"holding {b}" in current_state]
    if holding_blocks:
        b = holding_blocks[0]
        fixed_state = set(current_state) - {f"holding {b}"}
        fixed_state.update({f"{b} on table", f"clear {b}", "hand empty"})
        current_state = frozenset(fixed_state)

    return current_state

def generate_datasets(num_train=10000, num_eval=500):
    """Generates a blocksworld problems with optional harmless undo mistakes."""
    print("Generating 'Harmless Undo' dataset to teach recovery grammar...")

    def build_set(num_samples):
        # pylint: disable=too-many-locals
        data = []
        seen = set()
        while len(data) < num_samples:
            init_state = generate_random_state()
            goal_state = generate_random_state()

            if init_state == goal_state:
                continue
            relaxed_goal = set(cond for cond in goal_state if "on" in cond)
            optimal_path = bfs_solve(init_state, relaxed_goal)

            if not optimal_path or len(optimal_path) < 4:
                continue

            config_hash = (frozenset(init_state), frozenset(goal_state))
            if config_hash in seen:
                continue
            seen.add(config_hash)

            final_path = list(optimal_path)
            if random.random() < 0.30:
                # Pick a random step in the trajectory
                step_idx = random.randint(0, len(optimal_path) - 1)
                curr = set(init_state)
                for i in range(step_idx):
                    for act, nxt in get_valid_actions(curr):
                        if act == optimal_path[i]:
                            curr = nxt
                            break

                valid_moves = [act for act, _ in get_valid_actions(curr)]
                mistakes = [m for m in valid_moves if m != optimal_path[step_idx]]
                if mistakes:
                    mistake = random.choice(mistakes)
                    inverse = get_inverse_action(mistake)
                    final_path = optimal_path[:step_idx]+[mistake, inverse]+optimal_path[step_idx:]

            init_str = ", ".join(sorted(list(init_state)))
            goal_str = ", ".join(sorted(list(relaxed_goal)))
            prompt = f"[INIT] {init_str} [GOAL] {goal_str}"
            trajectory = ", ".join(final_path) + " [DONE]"

            data.append({"prompt": prompt, "trajectory": trajectory})
        return data

    train_data = build_set(num_train)
    eval_data = build_set(num_eval)

    with open("./data_cache/blocksworld_sub/train_big.jsonl", "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d) + "\n")
    with open("./data_cache/blocksworld_sub/eval_big.jsonl", "w", encoding="utf-8") as f:
        for d in eval_data:
            f.write(json.dumps(d) + "\n")
    print("Done! The model will now learn to Undo actions without shredding its grammar.")

if __name__ == "__main__":
    generate_datasets(num_train=50000, num_eval=1000)
