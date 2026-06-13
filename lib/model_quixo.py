import numpy as np
import lib.game_quixo as game_quixo
import lib.agents_quixo as agents_quixo
from numpy.typing import NDArray
from typing import Iterator

TUPLES = [
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19),
    (20, 21, 22, 23, 24),
    (0, 5, 10, 15, 20),
    (1, 6, 11, 16, 21),
    (2, 7, 12, 17, 22),
    (3, 8, 13, 18, 23),
    (4, 9, 14, 19, 24),
    (0, 6, 12, 18, 24),
    (4, 8, 12, 16, 20),
    (0, 1, 6, 11),
    (1, 2, 7, 6),
    (2, 3, 8, 13),
    (3, 4, 9, 8),
    (5, 6, 11, 16),
    (6, 7, 12, 11),
    (7, 8, 13, 18),
    (8, 9, 14, 13),
    (10, 11, 6, 7),
    (11, 12, 7, 8),
    (12, 13, 8, 9),
    (13, 14, 9, 8),
    (15, 16, 11, 6),
    (16, 17, 12, 11),
    (17, 18, 13, 12),
    (18, 19, 14, 13),
    (20, 21, 16, 11),
    (21, 22, 17, 16),
    (22, 23, 18, 17),
    (23, 24, 19, 18),
    (0, 5, 6, 11),
    (4, 9, 8, 13),
    (20, 15, 16, 11),
    (24, 19, 18, 13),
    (1, 6, 7, 12),
    (2, 7, 8, 13),
    (3, 8, 13, 18),
    (10, 5, 6, 7),
    (14, 9, 8, 7),
    (15, 10, 11, 12),
    (19, 14, 13, 12),
]


def symmetric_boards(state: NDArray[np.int8]) -> Iterator[NDArray[np.int8]]:
    """
    Generate basic Quixo symmetries.

    Args:
        state (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
    Yields:
        NDArray[np.int8]: A transformed board state (copy).
    """
    yield state
    yield np.rot90(state, 1)
    yield np.rot90(state, 2)
    yield np.rot90(state, 3)
    yield np.fliplr(state)
    yield np.flipud(state)


def map_tuple_index(flat_board: NDArray[np.int8], positions: tuple[int, ...]) -> int:
    """
    Map a base-3 encoded index for an n-tuple pattern on a board.

    Args:
        flat_board (NDArray[np.int8]): 1D array that represents a 5x5 Quixo board.
        positions (tuple[int, ...]): Tuple of indices into `flat_board` defining the n-tuple.
    Returns:
        int: Base-3 encoded index representing the pattern at those positions.
    """
    idx = 0
    for p in positions:
        cell = flat_board[p]
        if cell == -1:
            digit = 0
        elif cell == 0:
            digit = 1
        else:
            digit = 2
        idx = idx * 3 + digit

    return idx


class NTuple:
    def __init__(self, positions):
        self.positions = positions
        self.weights = np.zeros(3 ** len(positions), dtype=np.float32)
        self.trace = np.zeros_like(self.weights)


class NTupleNetwork:
    def __init__(self):
        self.tuples = [NTuple(t) for t in TUPLES]

    def evaluate(self, board):
        flat = board.reshape(-1)
        value = 0.0
        for tup in self.tuples:
            idx = map_tuple_index(flat, tup.positions)
            value += tup.weights[idx]
        return np.tanh(value)

    def update(self, board, delta, alpha, gamma=0.99, lam=0.7):
        flat = board.reshape(-1)
        for tup in self.tuples:
            tup.trace *= gamma * lam
            idx = map_tuple_index(flat, tup.positions)
            tup.trace[idx] += 1.0
            tup.weights += alpha * delta * tup.trace


def select_move_train(
    net: NTupleNetwork, state: NDArray[np.int8], cur_player: int, epsilon: float = 0.1
) -> int:
    """
    Selects a move using an epsilon-greedy policy with one-move lookahead evaluation.

    The function either explores randomly with probability epsilon or selects the move that
    maximises the network-evaluated value of the next state.

    Args:
        net (NTupleNetwork): N-tuple network that approximates the value function.
        state (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
        cur_player (int): Integer representing the player (1 or -1).
        epsilon (float): Probability of selecting a random legal move.

    Returns:
        int: Chosen action index.
    """
    legal_moves = game_quixo.possible_moves(state, cur_player)
    if len(legal_moves) == 0:
        raise ValueError("No legal moves available")
    if np.random.random() < epsilon:
        return int(np.random.choice(legal_moves))

    best_move = legal_moves[0]
    best_value = -999999

    for move in legal_moves:
        next_state, _ = game_quixo.move(state, move, cur_player)

        value = net.evaluate(next_state)
        if cur_player == -1:
            value = -value

        if value > best_value:
            best_value = value
            best_move = move

    return best_move


def play_game_train(
    net: NTupleNetwork, alpha: float = 0.01, gamma: float = 0.99, epsilon: float = 0.1
) -> tuple[int, int]:
    """
    Plays one self-play episode of Quixo using a TD-learning(λ) N-tuple network.

    Args:
        net (NTupleNetwork): N-tuple network that approximates the value function.
        alpha (float): Learning rate for TD updates.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Probability of selecting a random legal move.

    Returns:
        tuple[int, int]:
            - Game result from the perspective of player 1:
                1  → player 1 win
               -1  → player -1 win
                0  → draw
            - Number of moves played in the episode
    """
    state = game_quixo.INITIAL_STATE.copy()
    cur_player = int(np.random.choice([1, -1]))

    move_count = 0

    while True:
        action = select_move_train(net, state, cur_player, epsilon)

        if action is None:
            return 0, move_count

        next_state, won = game_quixo.move(state, action, cur_player)
        reward = 0.0
        if won == cur_player:
            reward = 1.0
        elif won == -cur_player:
            reward = -1.0
        terminal = won != 0

        value = net.evaluate(state) * cur_player

        if terminal:
            next_value = 0.0
        else:
            next_value = net.evaluate(next_state) * (-cur_player)

        target = reward + gamma * next_value
        delta = target - value

        for sym_state in symmetric_boards(state):
            net.update(sym_state, delta, alpha, gamma=gamma)

        if terminal:
            return won, move_count + 1

        state = next_state
        cur_player *= -1
        move_count += 1


def play_game_full(
    agent1: (
        agents_quixo.MCTSAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    ),
    agent2: (
        agents_quixo.MCTSAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    ),
    moves_before_tau_0: int,
    agent1_plays_first: bool | None = None,
):
    """
    Simulates a single self-play game between two neural networks using MCTS.

    Args:
        agent1: The first agent. Can be an MCTSAgent or a heuristic-based agent.
        agent2: The second agent. Can be an MCTSAgent or a heuristic-based agent.
        moves_before_tau_0 (int): Number of moves at the start of the game for which temperature parameter is 1, before
        switching to zero.
        agent1_plays_first (bool | None): If True, net1 goes first; if False, net2 goes first. If None, the first player
        is chosen randomly. Defaults to None.
    Returns:
        int: +1 if net1 wins, -1 if net2 wins, 0 for a draw.
        int: Total number of moves played in the game.
    """
    assert isinstance(moves_before_tau_0, int) and moves_before_tau_0 >= 0

    state = game_quixo.INITIAL_STATE
    agents = [agent1, agent2]
    if agent1_plays_first is None:
        cur_player = int(np.random.choice([1, -1]))
    else:
        cur_player = 1 if agent1_plays_first else -1
    move = 0
    tau = 1 if moves_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    while result is None:
        cur_player_idx = 0 if cur_player == 1 else 1
        agent = agents[cur_player_idx]

        if agent.name == "mcts_net":
            action, probs = agent.select_action(state, cur_player, tau)
        else:
            action = agent.select_action(state, cur_player)
            probs = np.zeros(game_quixo.N_ACTIONS, dtype=np.float32)
            probs[action] = 1.0

        game_history.append((state, cur_player, probs))

        state, won = game_quixo.move(state, action, cur_player)
        if won == cur_player:
            print(f"Game won by {cur_player}!")
            if cur_player == 1:
                net1_result = 1
            elif cur_player == -1:
                net1_result = -1
            break
        elif won == -cur_player:
            print(f"Game won by {-cur_player}!")
            if cur_player == 1:
                net1_result = -1
            elif cur_player == -1:
                net1_result = 1
            break
        cur_player = cur_player * -1
        if len(game_quixo.possible_moves(state, cur_player)) == 0:
            net1_result = 0
            break
        move += 1
        if move >= moves_before_tau_0:
            tau = 0

    return net1_result, move
