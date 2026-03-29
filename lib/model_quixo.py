import collections
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from lib import game_quixo, mcts_quixo


OBS_SHAPE = (3, game_quixo.SIZE, game_quixo.SIZE)
NUM_FILTERS = 64


class Net(nn.Module):
    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
        )

        body_out_shape = (NUM_FILTERS,) + input_shape[1:]

        self.conv_val = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1), nn.BatchNorm2d(1), nn.LeakyReLU()
        )
        conv_val_size = self._get_conv_val_size(body_out_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_val_size, 20), nn.LeakyReLU(), nn.Linear(20, 1), nn.Tanh()
        )

        self.conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 2, kernel_size=1), nn.BatchNorm2d(2), nn.LeakyReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(nn.Linear(conv_policy_size, actions_n))

    def _get_conv_val_size(self, shape):
        o = self.conv_val(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        v = self.conv_in(x)
        v = v + self.conv_1(v)
        v = v + self.conv_2(v)
        v = v + self.conv_3(v)
        v = v + self.conv_4(v)
        v = v + self.conv_5(v)
        val = self.conv_val(v)
        val = self.value(val.view(batch_size, -1))
        pol = self.conv_policy(v)
        pol = self.policy(pol.view(batch_size, -1))
        return pol, val


def encode_board_for_nn(dest_np: np.ndarray, state: NDArray[np.int8], player: int):
    """
    Encode a single board state into an array suitable for our neural network.

    Args:
        dest_np (np.ndarray): Target array of shape to store the neural network encoded state.
        state (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
        player (int): Integer representing the player (1 or -1).
    """
    assert dest_np.shape == OBS_SHAPE

    for col_idx, col in enumerate(state):
        for rev_row_idx, cell in enumerate(col):
            row_idx = game_quixo.SIZE - rev_row_idx - 1
            if cell == player:
                dest_np[0, row_idx, col_idx] = 1.0
            elif cell == -player:
                dest_np[1, row_idx, col_idx] = 1.0
            else:
                dest_np[2, row_idx, col_idx] = 1.0


def states_to_tensor_batch(
    state_list: list, player_list: list, device: str = "cpu"
) -> torch.Tensor:
    """
    Encodes states to shape used in neural network and returns a tensor, in batch form.

    Args:
        state_list (list): List of states.
        player_list (list): List of players.
        device (str): Device to run neural network inference on ("cpu" or "cuda"). Defaults to "cpu".
    Returns:
        torch.Tensor: PyTorch tensor batch of encoded board states.
    """
    assert isinstance(state_list, list)
    batch_size = len(state_list)
    batch = np.zeros((batch_size,) + OBS_SHAPE, dtype=np.float32)
    for idx, (state, player) in enumerate(zip(state_list, player_list)):
        encode_board_for_nn(batch[idx], state, player)
    return torch.tensor(batch).to(device)


def play_game(
    mcts_stores: mcts_quixo.MCTS | list[mcts_quixo.MCTS] | None,
    replay_buffer: collections.deque | None,
    net1: nn.Module,
    net2: nn.Module,
    steps_before_tau_0: int,
    n_iterations: int,
    n_simulations: int,
    net1_plays_first: bool | None = None,
    device: str = "cpu",
):
    """
    Simulates a single self-play game between two neural networks using MCTS.

    Args:
        mcts_stores (mcts_quixo.MCTS | list[mcts_quixo.MCTS] | None): A singular Monte-Carlo Tree Search class or list
        of Monte-Carlo Tree Search classes that keeps statistics for every state encountered during the search. Can be
        set as None to create a new instance.
        replay_buffer (collections.deque | None): Replay buffer storing (state, player, policy_probs, result) tuples.
        Can be set as None to disable replay storage.
        net1 (nn.Module): Neural network that predicts policy and value.
        net2 (nn.Module): Neural network that predicts policy and value.
        steps_before_tau_0 (int): Number of moves at the start of the game for which temperature parameter is 1, before
        switching to zero.
        n_iterations (int): Number of times MCTS batch of simulations are run.
        n_simulations (int): Number of MCTS simulations to batch together.
        net1_plays_first (bool | None): If True, net1 goes first; if False, net2 goes first. If None, the first player
        is chosen randomly. Defaults to None.
        device (str): Device to run neural network inference on ("cpu" or "cuda"). Defaults to "cpu".
    Returns:
        int: +1 if net1 wins, -1 if net2 wins, 0 for a draw.
        int: Total number of moves played in the game.
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mcts_quixo.MCTS, type(None), list))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(n_iterations, int) and n_iterations > 0
    assert isinstance(n_simulations, int) and n_simulations > 0

    if mcts_stores is None:
        mcts_stores = [mcts_quixo.MCTS(), mcts_quixo.MCTS()]
    elif isinstance(mcts_stores, mcts_quixo.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = game_quixo.encode_board(game_quixo.INITIAL_STATE)
    nets = [net1, net2]
    if net1_plays_first is None:
        cur_player = np.random.choice([1, -1])
    else:
        cur_player = 1 if net1_plays_first else -1
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    while result is None:
        cur_player_idx = 0 if cur_player == 1 else 1
        print(
            f"Move number is {step} and current_player is {cur_player} and current_player_idx is {cur_player_idx}"
        )
        mcts_stores[cur_player_idx].run_mcts(
            n_iterations,
            n_simulations,
            state,
            cur_player,
            nets[cur_player_idx],
            device=device,
        )
        probs, _ = mcts_stores[cur_player_idx].get_policy_value(state, tau=tau)
        game_history.append((state, cur_player, probs))

        print(f"Probs are: {probs}")
        legal_moves = game_quixo.possible_moves(state, cur_player)
        mask = np.zeros(game_quixo.N_ACTIONS)
        mask[legal_moves] = 1
        legal_probs = probs * mask
        if legal_probs.sum() == 0:
            action = np.random.choice(legal_moves)
        else:
            legal_probs = legal_probs / np.sum(legal_probs)
            action = np.random.choice(game_quixo.N_ACTIONS, p=legal_probs)
        print(f"Legal probs are: {legal_probs}")

        print(f"Model action chosen is {action}")
        if action not in game_quixo.possible_moves(state, cur_player):
            print("Impossible action selected")
        state, won = game_quixo.move(state, action, cur_player)
        if won:
            print(f"Game won by {cur_player}!")
            result = 1
            net1_result = 1 if cur_player == 1 else -1
            break
        cur_player = cur_player * -1
        if len(game_quixo.possible_moves(state, cur_player)) == 0:
            result = 0
            net1_result = 0
            break
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            replay_buffer.append((state, cur_player, probs, result))
            result = -result

    return net1_result, step
