import math as m
from typing import Any

import numpy as np
import lib.game_quixo as game_quixo
import lib.model_quixo as model_quixo


class MCTS:
    """
    Monte-Carlo Tree Search class that keeps statistics for every state encountered during the search.

    Attributes:
        c_puct (float): Exploration constant.
        visit_count (dict): Dictionary to store visit counts.
        value (dict): Dictionary to store values.
        value_avg (dict): Dictionary to store average values.
        probs (dict): Dictionary to store probabilities.
    """

    def __init__(self, c_puct: float = 1.0):
        self.c_puct = c_puct
        self.visit_count = {}
        self.value = {}
        self.value_avg = {}
        self.probs = {}

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def is_leaf(self, state_int: int, player: int) -> bool:
        return (state_int, player) not in self.probs

    def find_leaf(
        self, state_int: int, player: int
    ) -> tuple[float | None, int, int | Any, list[int], list[int], list[int]]:
        """
        Traverse the tree from the root until a leaf node is found.

        Args:
            state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
            player (int): Integer representing the player (1 or -1).
        Returns:
            float | None: Value of the game outcome for the current player at the leaf node. None if not terminal.
            int: Encoded integer that represents a unique Quixo 5x5 board state.
            int: Integer representing the player (1 or -1).
            list: List of visited states.
            list: List of implemented actions.
            list: List of players who carried out actions.
        """
        assert player in [game_quixo.PLAYER_O, game_quixo.PLAYER_X]

        states_tracked = []
        actions_tracked = []
        players_tracked = []
        cur_state = state_int
        cur_player = player
        value = None
        visited = set()
        # Keep searching until a leaf node is found
        while not self.is_leaf(cur_state, cur_player):
            # If we revisit the same state, then treat it as a draw and break early
            if (cur_state, cur_player) in visited:
                value = 0.0
                break
            visited.add((cur_state, cur_player))

            states_tracked.append(cur_state)
            players_tracked.append(cur_player)

            key = (cur_state, cur_player)

            counts = self.visit_count[key]
            total_sqrt = m.sqrt(1 + sum(counts))
            probs = self.probs[key]
            values_avg = self.value_avg[key]

            # Calculate PUCT score
            score = [
                mcts_value_avg + self.c_puct * mcts_prob * total_sqrt / (1 + mcts_count)
                for mcts_value_avg, mcts_prob, mcts_count in zip(
                    values_avg, probs, counts
                )
            ]
            invalid_actions = set(range(game_quixo.N_ACTIONS)) - set(
                game_quixo.possible_moves(
                    game_quixo.decode_board(cur_state), cur_player
                )
            )
            for invalid in invalid_actions:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            actions_tracked.append(action)
            # Transition to the next state using best action
            cur_state, won = game_quixo.move(
                game_quixo.decode_board(cur_state), action, cur_player
            )
            if won == cur_player:
                value = 1.0
                break
            elif won == -cur_player:
                value = -1.0
                break
            cur_player *= -1
            moves_count = len(game_quixo.possible_moves(cur_state, cur_player))
            # If no moves left, then it is a draw
            if value is None and moves_count == 0:
                value = 0.0
            cur_state = game_quixo.encode_board(cur_state)

        return (
            value,
            cur_state,
            cur_player,
            states_tracked,
            actions_tracked,
            players_tracked,
        )

    def run_mcts(
        self,
        n_simulations: int,
        state_int: int,
        cur_player: int,
        net: model_quixo.NTupleNetwork,
    ):
        """
        Perform multiple MCTS simulations.

        Args:
            n_simulations (int): Number of MCTS simulations to run.
            state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
            cur_player (int): Integer representing the player (1 or -1).
            net (model_quixo.NTupleNetwork): TD network that predicts value.
        """
        # Find leaf nodes for multiple MCTS simulations and store for backup, expand, or discard if already seen leaf
        # state to be expanded
        for sim_idx in range(n_simulations):
            (
                value,
                leaf_state_int,
                leaf_player,
                states_tracked,
                actions_tracked,
                players_tracked,
            ) = self.find_leaf(state_int, cur_player)

            # Leaf expansion and evaluation
            if value is None:
                leaf_state = game_quixo.decode_board(leaf_state_int)

                # Evaluate leaf
                value = net.value_function_for_player(leaf_state, leaf_player)

                key = (leaf_state_int, leaf_player)
                legal_moves = game_quixo.possible_moves(leaf_state, leaf_player)

                action_values = []
                for action in legal_moves:
                    next_state, won = game_quixo.move(
                        leaf_state.copy(), action, leaf_player
                    )

                    if won == leaf_player:
                        action_value = 1.0
                    elif won == -leaf_player:
                        action_value = -1.0
                    else:
                        action_value = net.value_function_for_player(
                            next_state, leaf_player
                        )
                    action_values.append(action_value)

                temperature = 1.0
                values = np.array(action_values)
                values = values - values.max()
                prior_values = np.exp(values / temperature)
                prior = np.zeros(game_quixo.N_ACTIONS, dtype=np.float32)
                prior[legal_moves] = prior_values / prior_values.sum()

                self.visit_count[key] = [0] * game_quixo.N_ACTIONS
                self.value[key] = [0.0] * game_quixo.N_ACTIONS
                self.value_avg[key] = [0.0] * game_quixo.N_ACTIONS
                self.probs[key] = prior

            # Backup this simulation immediately
            for state, action, player in zip(
                states_tracked[::-1], actions_tracked[::-1], players_tracked[::-1]
            ):
                key = (state, player)

                self.visit_count[key][action] += 1
                self.value[key][action] += value
                self.value_avg[key][action] = (
                    self.value[key][action] / self.visit_count[key][action]
                )

                value = -value

    def get_policy_value(
        self, state_int: int, cur_player: int, tau: int = 1
    ) -> tuple[list[float], list[float]]:
        """
        Convert MCTS search statistics at a state into a policy distribution and values for each action.

        Args:
            state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
            cur_player (int): Integer representing the player (1 or -1).
            tau (int): Parameter that controls degree of exploration for returned policy. Defaults to 1.
        Returns:
            list[float]: Policy (π(s)) for a given state.
            list[float]:: Values (Q(s,a)) for a given state.
        """
        key = (state_int, cur_player)
        counts = self.visit_count[key]
        if tau == 0:
            probs = [0.0] * game_quixo.N_ACTIONS
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[key]
        return probs, values
