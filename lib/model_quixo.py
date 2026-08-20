import numpy as np
import lib.game_quixo as game_quixo
import lib.agents_quixo as agents_quixo
from numpy.typing import NDArray
from typing import Iterator
from numba import njit
import random


GREEDY_WIN_AGENT_PROB = 0.25


def generate_random_walk(length, rng):
    neighbours = {}

    for r in range(5):
        for c in range(5):
            idx = r * 5 + c
            adjacent = []

            for dr, dc in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                rr = r + dr
                cc = c + dc

                if 0 <= rr < 5 and 0 <= cc < 5:
                    adjacent.append(rr * 5 + cc)

            neighbours[idx] = adjacent

    start = rng.randint(0, 24)
    walk = [start]

    while len(walk) < length:
        current = walk[-1]

        candidates = [
            position for position in neighbours[current] if position not in walk
        ]

        if not candidates:
            return None

        walk.append(rng.choice(candidates))

    return tuple(walk)


def generate_tuples(
    n_random_6=800,
    n_random_7=1200,
    seed=42,
):
    tuples = set()

    def add(t):
        tuples.add(tuple(t))

    for r in range(5):
        add([r * 5 + i for i in range(5)])

    for c in range(5):
        add([c + 5 * i for i in range(5)])

    add([i * 6 for i in range(5)])
    add([4 + i * 4 for i in range(5)])

    for r in range(3):
        for c in range(3):
            base = r * 5 + c
            add(
                [
                    base,
                    base + 1,
                    base + 2,
                    base + 5,
                    base + 6,
                    base + 7,
                    base + 10,
                    base + 11,
                    base + 12,
                ]
            )

    for r in range(4):
        for c in range(4):
            base = r * 5 + c
            add([base, base + 1, base + 5, base + 6])

    offsets = [
        (0, 6, 12, 18),
        (2, 6, 10, 14),
        (0, 4, 20, 24),
    ]

    for pattern in offsets:
        add(pattern)

    rng = np.random.RandomState(seed)
    target_random = n_random_6 + n_random_7
    while len(tuples) < 40 + target_random:
        if len(tuples) < 40 + n_random_6:
            length = 6
        else:
            length = 7

        walk = generate_random_walk(length, rng)
        if walk is not None:
            add(walk)

    return sorted(tuples)


TUPLES = generate_tuples(
    n_random_6=150,
    n_random_7=250,
    seed=42,
)


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


@njit(cache=True)
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


def perspective_state(
    state: NDArray[np.int8],
    cur_player: int,
) -> NDArray[np.int8]:
    """
    Convert a board into the perspective of `player`.

    The network always evaluates positions with the current player
    represented by +1.

    Args:
        state (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
        cur_player (int): Integer representing the player (1 or -1).
    Returns:
        Board represented from the current player's perspective.
    """
    assert cur_player in (game_quixo.PLAYER_X, game_quixo.PLAYER_O)

    if cur_player == game_quixo.PLAYER_X:
        return state.copy()

    return (-state).astype(np.int8)


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
            nonzero = tup.trace != 0
            tup.weights[nonzero] += alpha * delta * tup.trace[nonzero]

    def reset_traces(self):
        for tup in self.tuples:
            tup.trace.fill(0)

    def evaluate_for_player(
        self,
        board: NDArray[np.int8],
        cur_player: int,
    ) -> float:
        """
        Evaluate a board from the perspective of `player`.

        The network always sees the current player as +1.
        """
        board_perspective = perspective_state(board, cur_player)
        return self.evaluate(board_perspective)


def select_move_train(
    net: NTupleNetwork, state: NDArray[np.int8], cur_player: int, epsilon_decay: float
) -> int:
    """
    Selects a move using an epsilon-greedy approach with one-move lookahead evaluation.

    The function either explores randomly with probability epsilon or selects the move that
    maximises the network-evaluated value of the next state.

    Args:
        net (NTupleNetwork): N-tuple network that approximates the value function.
        state (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
        cur_player (int): Integer representing the player (1 or -1).
        epsilon_decay (float): Decay factor to determine the probability of selecting a random legal move.
    Returns:
        int: Chosen action index.
    """
    epsilon = max(0.01, 0.2 * epsilon_decay)

    legal_moves = game_quixo.possible_moves(state, cur_player)
    if len(legal_moves) == 0:
        raise ValueError("No legal moves available")
    if np.random.random() < epsilon:
        return int(np.random.choice(legal_moves))

    best_move = legal_moves[0]
    best_value = -999999 * cur_player

    for move in legal_moves:
        next_state, _ = game_quixo.move(state, move, cur_player)

        value = net.evaluate(next_state)

        if cur_player == 1:
            if value > best_value:
                best_value = value
                best_move = move
        if cur_player == -1:
            if value < best_value:
                best_value = value
                best_move = move

    return best_move


def play_game_train(
    net: NTupleNetwork,
    alpha: float = 0.2,
    gamma: float = 0.99,
    epsilon_decay: float = 1,
) -> tuple[int, int]:
    """
    Plays one self-play episode of Quixo using a TD-learning(λ) N-tuple network.

    Args:
        net (NTupleNetwork): N-tuple network that approximates the value function.
        alpha (float): Learning rate for TD updates.
        gamma (float): Discount factor for future rewards.
        epsilon_decay (float): Decay factor to determine the probability of selecting a random legal move.
    Returns:
        tuple[int, int]:
            - Game result from the perspective of player 1:
                1  → player 1 win
               -1  → player -1 win
                0  → draw
            - Number of moves played in the episode
    """
    net.reset_traces()
    state = game_quixo.INITIAL_STATE.copy()
    cur_player = int(np.random.choice([1, -1]))
    move_count = 0
    player_x_last_state = None

    while True:
        if cur_player == -1 and (random.random() <= GREEDY_WIN_AGENT_PROB):
            action = agents_quixo.GreedyWinAgent.select_action(state, cur_player)
        else:
            action = select_move_train(net, state, cur_player, epsilon_decay)
        next_state, won = game_quixo.move(state, action, cur_player)
        move_count += 1

        reward = 0.0
        terminal = won != 0

        if cur_player == 1:
            if player_x_last_state:
                last_value = net.evaluate_for_player(player_x_last_state, cur_player)

                if terminal:
                    reward = won
                    next_value = 0.0
                else:
                    next_value = net.evaluate_for_player(next_state, cur_player)

                delta = (reward + gamma * next_value) - last_value
                net.update(player_x_last_state, delta, alpha, gamma=gamma)

            player_x_last_state = next_state

        if terminal:
            return won, move_count + 1

        state = next_state
        cur_player *= -1


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
