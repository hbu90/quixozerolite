import math as m
from typing import Any

import numpy as np
import torch

import lib.game_quixo as game_quixo
import lib.model_quixo as model_quixo

import torch.nn.functional as functional
import torch.nn as nn

DIRICHLET_EPSILON = 0.25


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

    def is_leaf(self, state_int: int) -> bool:
        return state_int not in self.probs

    def find_leaf(
        self, state_int: int, player: int
    ) -> tuple[float | None, int, int | Any, list[int], list[int]]:
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
        """
        assert player in [game_quixo.PLAYER_O, game_quixo.PLAYER_X]

        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None
        visited = set()
        # Keep searching until a leaf node is found
        while not self.is_leaf(cur_state):
            # If we revisit the same state, then treat it as a draw and break early
            if (cur_state, cur_player) in visited:
                value = 0.0
                break
            visited.add((cur_state, cur_player))

            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = m.sqrt(1 + sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # Only apply this random noise to the probability at the root node
            if cur_state == state_int:
                noises = np.random.dirichlet([0.03] * game_quixo.N_ACTIONS)
                probs = [
                    (1 - DIRICHLET_EPSILON) * prob + DIRICHLET_EPSILON * noise
                    for prob, noise in zip(probs, noises)
                ]
            # Calculate PUCT score
            score = [
                value + self.c_puct * prob * total_sqrt / (1 + count)
                for value, prob, count in zip(values_avg, probs, counts)
            ]
            invalid_actions = set(range(game_quixo.N_ACTIONS)) - set(
                game_quixo.possible_moves(cur_state, cur_player)
            )
            for invalid in invalid_actions:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            actions.append(action)
            # Transition to the next state using best action
            cur_state, won = game_quixo.move(cur_state, action, cur_player)
            if won == cur_player:
                value = 1.0
                break
            elif won == -cur_player:
                value = -1.0
                break
            cur_player = cur_player * -1
            moves_count = len(game_quixo.possible_moves(cur_state, cur_player))
            # If no moves left, then it is a draw
            if value is None and moves_count == 0:
                value = 0.0

        return value, cur_state, cur_player, states, actions

    def run_mcts(
        self,
        n_iterations: int,
        n_simulations: int,
        state_int: int,
        player: int,
        net: nn.Module,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Run MCTS simulations from a given game state.

        Args:
            n_iterations (int): Number of times MCTS batch of simulations are run.
            n_simulations (int): Number of MCTS simulations to batch together.
            state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
            player (int): Integer representing the player (1 or -1).
            net (nn.Module): Neural network that predicts policy and value.
            device (torch.device): Device to run neural network inference on ("cpu" or "cuda"). Defaults to "cpu".
        """
        for step in range(n_iterations):
            self.mcts_simulations_batch(n_simulations, state_int, player, net, device)

    def mcts_simulations_batch(
        self,
        n_simulations: int,
        state_int: int,
        player: int,
        net: nn.Module,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Perform multiple MCTS simulations per call.

        Args:
            n_simulations (int): Number of MCTS simulations to batch together.
            state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
            player (int): Integer representing the player (1 or -1).
            net (nn.Module): Neural network that predicts policy and value.
            device (torch.device): Device to run neural network inference on ("cpu" or "cuda"). Defaults to "cpu".
        """
        backup_queue = []
        expand_states = []  # States to be evaluated by neural network
        expand_players = []
        expand_queue = []
        planned = set()

        # Find leaf nodes for multiple MCTS simulations and store for backup, expand, or discard if already seen leaf
        # state to be expanded
        for sim_idx in range(n_simulations):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(
                state_int, player
            )
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    leaf_state_lists = game_quixo.decode_board(leaf_state)
                    expand_states.append(leaf_state_lists)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states, actions))

        # Expand nodes using neural network
        if expand_queue:
            batch_v = model_quixo.states_to_tensor_batch(
                expand_states, expand_players, device
            )
            logits_v, values_v = net(batch_v)
            probs_v = functional.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            # Create nodes
            for (leaf_state, states, actions), value, prob in zip(
                expand_queue, values, probs
            ):
                self.visit_count[leaf_state] = [0] * game_quixo.N_ACTIONS
                self.value[leaf_state] = [0.0] * game_quixo.N_ACTIONS
                self.value_avg[leaf_state] = [0.0] * game_quixo.N_ACTIONS

                legal_moves = game_quixo.possible_moves(leaf_state, leaf_player)
                mask = np.zeros(game_quixo.N_ACTIONS)
                mask[legal_moves] = 1
                prob = prob * mask
                if prob.sum() > 0:
                    prob /= prob.sum()
                else:
                    prob = mask / mask.sum()
                self.probs[leaf_state] = prob

                backup_queue.append((value, states, actions))

        # Backup of searches
        for value, states, actions in backup_queue:
            # value is from perspective of leaf_player

            for state_int, action in zip(states[::-1], actions[::-1]):
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += value
                self.value_avg[state_int][action] = (
                    self.value[state_int][action] / self.visit_count[state_int][action]
                )
                value = -value  # TO-DO: Look at this logic

    def get_policy_value(
        self, state_int: int, tau: int = 1
    ) -> tuple[list[float], float]:
        """
        Convert MCTS search statistics at a state into a policy distribution and values for each action.

        Args:
            state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
            tau (int): Parameter that controls degree of exploration for returned policy. Defaults to 1.
        Returns:
            list[float]: Policy (π(s)) for a given state.
            float: Values (Q(s,a)) for a given state.
        """
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * game_quixo.N_ACTIONS
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[state_int]
        return probs, values
