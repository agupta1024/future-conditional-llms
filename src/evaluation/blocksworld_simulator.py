""" Blocksworld Simulator for symbolic state tracking """

class BWSimulator:
    """A strict symbolic state tracker for Blocksworld."""
    def __init__(self, init_str):
        # self.state = set(init_str.strip().split(", "))
        self.state = {cond.strip() for cond in init_str.split(",") if cond.strip()}
        self.is_valid = True

    def apply_action(self, action): #pylint: disable=too-many-branches, too-many-return-statements
        """Apply a single action to the current state, updating it if valid."""
        action = action.strip()
        # if not action or action == "[DONE]" or action.startswith("<|endoftext|>"):
            # return False # End of execution
        parts = action.split()
        if len(parts) == 0:
            self.is_valid = False
            return False
        cmd = parts[0]

        if cmd == "pickup":
            if len(parts) < 2:
                self.is_valid = False
                return False
            b = parts[1]
            reqs = {f"clear {b}", f"{b} on table", "hand empty"}
            if not reqs.issubset(self.state):
                self.is_valid = False
                return False
            self.state -= reqs
            self.state.add(f"holding {b}")

        elif cmd == "putdown":
            if len(parts) < 2:
                self.is_valid = False
                return False
            b = parts[1]
            if f"holding {b}" not in self.state:
                self.is_valid = False
                return False
            self.state.remove(f"holding {b}")
            self.state.update({f"{b} on table", f"clear {b}", "hand empty"})

        elif cmd == "unstack":
            if len(parts) < 3:
                self.is_valid = False
                return False
            b, c = parts[1], parts[2]
            reqs = {f"clear {b}", f"{b} on {c}", "hand empty"}
            if not reqs.issubset(self.state):
                self.is_valid = False
                return False
            self.state -= reqs
            self.state.update({f"holding {b}", f"clear {c}"})

        elif cmd == "stack":
            if len(parts) < 3:
                self.is_valid = False
                return False
            b, c = parts[1], parts[2]
            reqs = {f"holding {b}", f"clear {c}"}
            if not reqs.issubset(self.state):
                self.is_valid = False
                return False
            self.state -= reqs
            self.state.update({f"{b} on {c}", f"clear {b}", "hand empty"})
        else:
            self.is_valid = False
            return False # Hallucinated gibberish

        return True

    def score_goal(self, goal_str):
        """Check if the current state satisfies the goal conditions."""
        goal_conds = set(goal_str.strip().split(", "))
        satisfied = len(goal_conds.intersection(self.state))

        return {
            "success": goal_conds.issubset(self.state),
            "partial_pct": (satisfied / len(goal_conds)) * 100 if goal_conds else 0.0,
            "legal_execution": self.is_valid
        }
