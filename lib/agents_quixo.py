import random
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from lib import game_quixo


@dataclass
class RandomAgent:
    name: str = "random"

    @staticmethod
    def select_action(state: NDArray[np.int8], player: int) -> int:
        legal_moves = game_quixo.possible_moves(state, player)
        return int(random.choice(legal_moves))


@dataclass
class GreedyWinAgent:
    """
    If any move wins immediately, play it.
    Otherwise, play a random legal move.
    """

    name: str = "greedy_win"

    @staticmethod
    def select_action(state: NDArray[np.int8], player: int) -> int:
        legal_moves = game_quixo.possible_moves(state, player)

        for action in legal_moves:
            new_state, won = game_quixo.move(state, action, player)
            if won == player:
                return action

        return int(random.choice(legal_moves))


@dataclass
class WinBlockAgent:
    """
    Priority:
    1. If agent can win immediately -> win
    2. Else if opponent can win next move -> block it
    3. Else random move
    """

    name: str = "win_block"

    @staticmethod
    def select_action(state: NDArray[np.int8], player: int) -> int:
        legal_moves = game_quixo.possible_moves(state, player)

        for action in legal_moves:
            new_state, won = game_quixo.move(state, action, player)
            if won == player:
                return action

        opp = -player
        for action in legal_moves:
            new_state, won = game_quixo.move(state, action, player)

            opp_legal = game_quixo.possible_moves(new_state, opp)
            for opp_action in opp_legal:
                _, opp_won = game_quixo.move(new_state, opp_action, opp)
                if opp_won == opp:
                    return action

        return int(random.choice(legal_moves))


class MCTSAgent:
    def __init__(self, net, mcts, n_iters, n_sims):
        self.net = net
        self.mcts = mcts
        self.n_iters = n_iters
        self.n_sims = n_sims
        self.name = "mcts_net"

    def select_action(
        self, state: NDArray[np.int8], player: int, tau: float = 1.0
    ) -> tuple[int, np.ndarray]:

        state_copy = state.copy()
        player_copy = player

        # Flip board due to way TD network was trained
        if player == -1:
            state_copy *= -1
            player_copy *= -1

        state_int = game_quixo.encode_board(state_copy)

        self.mcts.run_mcts(
            n_iterations=self.n_iters,
            n_simulations=self.n_sims,
            state_int=state_int,
            player=player_copy,
            net=self.net,
        )

        probs, _ = self.mcts.get_policy_value(state_int, tau=tau)

        legal_moves = game_quixo.possible_moves(state_copy, player_copy)
        mask = np.zeros(game_quixo.N_ACTIONS)
        mask[legal_moves] = 1
        probs = np.array(probs, dtype=np.float32)

        legal_probs = probs * mask
        if legal_probs.sum() == 0:
            action = int(np.random.choice(legal_moves))
        else:
            legal_probs /= legal_probs.sum()
            action = int(np.random.choice(game_quixo.N_ACTIONS, p=legal_probs))

        return action, probs
