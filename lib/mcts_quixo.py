"""
Monte-Carlo Tree Search
"""
import math as m
import numpy as np

from lib import game_quixo, model_quixo

import torch.nn.functional as F


class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """
    def __init__(self, c_puct=1.0):
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

    def is_leaf(self, state_int):
        return state_int not in self.probs

    def find_leaf(self, state_int, player):
        """
        """
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None
        visited = set()
        print("Find leaf")
        while not self.is_leaf(cur_state):
            print(f"Current state is : {cur_state}")
            if (cur_state, cur_player) in visited:
                value = 0.0
                break
            visited.add((cur_state, cur_player))

            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = m.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # choose action to take, in the root node add the Dirichlet noise to the probs
            if cur_state == state_int:
                noises = np.random.dirichlet(
                    [0.03] * game_quixo.N_ACTIONS)
                probs = [
                    0.75 * prob + 0.25 * noise
                    for prob, noise in zip(probs, noises)
                ]
            score = [
                value + self.c_puct*prob*total_sqrt/(1+count)
                for value, prob, count in
                    zip(values_avg, probs, counts)
            ]
            print(f"MCTS score: {score}")
            invalid_actions = set(range(game_quixo.N_ACTIONS)) - \
                              set(game_quixo.possible_moves(cur_state, cur_player))
            print(f"MCTS invalid actions: {invalid_actions}")
            for invalid in invalid_actions:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            print(f"MCTS search action:{action}")
            actions.append(action)
            cur_state, won = game_quixo.move(
                cur_state, action, cur_player)
            print(f"New MCTS cur_state: {cur_state}, won: {won}, cur_player: {cur_player}")
            if won:
                # if somebody won the game, the value of the final state is -1 (as it is on opponent's turn)
                value = -1.0
            cur_player = cur_player*-1
            # check for the draw
            moves_count = len(game_quixo.possible_moves(cur_state, cur_player))
            if value is None and moves_count == 0:
                value = 0.0

        return value, cur_state, cur_player, states, actions

    def search_batch(self, mcts_searches, batch_size, state_int,
                     player, net, device="cpu"):
        print(f"Initial state is {state_int}")
        for step in range(mcts_searches):
            print(f"MCTS search number is {step}")

            self.search_minibatch(batch_size, state_int,
                                  player, net, device)

    def search_minibatch(self, count, state_int, player,
                         net, device="cpu"):
        """
        Perform several MCTS searches.
        """
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()
        for step in range(count):
            print(f"MCTS mini-batch number is {step}")
            value, leaf_state, leaf_player, states, actions = \
                self.find_leaf(state_int, player)
            print(f"New leaf state is {leaf_state}")
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    leaf_state_lists = game_quixo.decode_board(
                        leaf_state)
                    expand_states.append(leaf_state_lists)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states,
                                         actions))

        # do expansion of nodes
        if expand_queue:
            batch_v = model_quixo.state_lists_to_batch(
                expand_states, expand_players, device)
            logits_v, values_v = net(batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            # create the nodes
            for (leaf_state, states, actions), value, prob in \
                    zip(expand_queue, values, probs):
                self.visit_count[leaf_state] = [0]*game_quixo.N_ACTIONS
                self.value[leaf_state] = [0.0]*game_quixo.N_ACTIONS
                self.value_avg[leaf_state] = [0.0]*game_quixo.N_ACTIONS
                # self.probs[leaf_state] = prob
                legal_moves = game_quixo.possible_moves(leaf_state, player)
                mask = np.zeros(game_quixo.N_ACTIONS)
                mask[legal_moves] = 1
                prob = prob * mask
                if prob.sum() > 0:
                    prob /= prob.sum()
                else:
                    prob = mask / mask.sum()
                self.probs[leaf_state] = prob

                backup_queue.append((value, states, actions))

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state_int, action in zip(states[::-1],
                                         actions[::-1]):
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += cur_value
                self.value_avg[state_int][action] = \
                    self.value[state_int][action] / \
                    self.visit_count[state_int][action]
                cur_value = -cur_value

    def get_policy_value(self, state_int, tau=1):
        """
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
