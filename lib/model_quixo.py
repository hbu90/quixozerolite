import collections
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
import lib.game_quixo as game_quixo
import lib.agents_quixo as agents_quixo

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

    for row_idx, row in enumerate(state):
          for col_idx, cell in enumerate(row):
              if cell == player:
                  dest_np[0, row_idx, col_idx] = 1.0
              elif cell == -player:
                  dest_np[1, row_idx, col_idx] = 1.0
              else:
                  dest_np[2, row_idx, col_idx] = 1.0


def states_to_tensor_batch(
    state_list: list,
    player_list: list,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Encodes states to shape used in neural network and returns a tensor, in batch form.

    Args:
        state_list (list): List of states.
        player_list (list): List of players.
        device (torch.device): Device to run neural network inference on ("cpu" or "cuda"). Defaults to "cpu".
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
    replay_buffer: collections.deque | None,
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
    steps_before_tau_0: int,
    agent1_plays_first: bool | None = None,
):
    """
    Simulates a single self-play game between two neural networks using MCTS.

    Args:
        replay_buffer (collections.deque | None): Replay buffer storing (state, player, policy_probs, result) tuples.
        Can be set as None to disable replay storage.
        agent1: The first agent. Can be an MCTSAgent or a heuristic-based agent.
        agent2: The second agent. Can be an MCTSAgent or a heuristic-based agent.
        steps_before_tau_0 (int): Number of moves at the start of the game for which temperature parameter is 1, before
        switching to zero.
        agent1_plays_first (bool | None): If True, net1 goes first; if False, net2 goes first. If None, the first player
        is chosen randomly. Defaults to None.
    Returns:
        int: +1 if net1 wins, -1 if net2 wins, 0 for a draw.
        int: Total number of moves played in the game.
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0

    state = game_quixo.encode_board(game_quixo.INITIAL_STATE)
    agents = [agent1, agent2]
    if agent1_plays_first is None:
        cur_player = np.random.choice([1, -1])
    else:
        cur_player = 1 if agent1_plays_first else -1
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
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
            result = won
            if cur_player == 1:
                net1_result = 1
            elif cur_player == -1:
                net1_result = -1
            break
        elif won == -cur_player:
            print(f"Game won by {-cur_player}!")
            result = -won
            if cur_player == 1:
                net1_result = -1
            elif cur_player == -1:
                net1_result = 1
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
