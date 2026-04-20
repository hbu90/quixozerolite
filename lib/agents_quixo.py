import random
from dataclasses import dataclass
import numpy as np
from lib import game_quixo


@dataclass
class RandomAgent:
    name: str = "random"

    @staticmethod
    def select_action(state_int: int, player: int) -> int:
        legal_moves = game_quixo.possible_moves(state_int, player)
        return random.choice(legal_moves)


@dataclass
class GreedyWinAgent:
    """
    If any move wins immediately, play it.
    Otherwise, play a random legal move.
    """

    name: str = "greedy_win"

    @staticmethod
    def select_action(state_int: int, player: int) -> int:
        legal_moves = game_quixo.possible_moves(state_int, player)

        for action in legal_moves:
            new_state, won = game_quixo.move(state_int, action, player)
            if won == player:
                return action

        return random.choice(legal_moves)


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
    def select_action(state_int: int, player: int) -> int:
        legal_moves = game_quixo.possible_moves(state_int, player)

        for action in legal_moves:
            new_state, won = game_quixo.move(state_int, action, player)
            if won == player:
                return action

        opp = -player
        for action in legal_moves:
            new_state, won = game_quixo.move(state_int, action, player)

            opp_legal = game_quixo.possible_moves(new_state, opp)
            for opp_action in opp_legal:
                _, opp_won = game_quixo.move(new_state, opp_action, opp)
                if opp_won == opp:
                    return action

        return random.choice(legal_moves)


class MCTSAgent:
    def __init__(self, net, mcts, n_iters, n_sims, device="cpu"):
        self.net = net
        self.mcts = mcts
        self.n_iters = n_iters
        self.n_sims = n_sims
        self.device = device
        self.name = "mcts_net"

    def select_action(
        self, state_int: int, player: int, tau: float = 1.0
    ) -> tuple[int, list[float]]:
        self.mcts.clear()

        self.mcts.run_mcts(
            n_iterations=self.n_iters,
            n_simulations=self.n_sims,
            state_int=state_int,
            player=player,
            net=self.net,
            device=self.device,
        )

        probs, _ = self.mcts.get_policy_value(state_int, tau=tau)

        legal_moves = game_quixo.possible_moves(state_int, player)
        mask = np.zeros(game_quixo.N_ACTIONS)
        mask[legal_moves] = 1

        legal_probs = probs * mask
        if legal_probs.sum() == 0:
            action = np.random.choice(legal_moves)
        else:
            legal_probs /= legal_probs.sum()
            action = np.random.choice(game_quixo.N_ACTIONS, p=legal_probs)

        return action, probs
