import collections
import numpy as np
import pytest
import torch

from lib import game_quixo
from lib import model_quixo


@pytest.fixture
def empty_board() -> np.ndarray:
    return np.zeros((game_quixo.SIZE, game_quixo.SIZE), dtype=np.int8)


@pytest.fixture
def sample_board() -> np.ndarray:
    # board contains player=1, opponent=-1, and empty=0
    board = np.zeros((game_quixo.SIZE, game_quixo.SIZE), dtype=np.int8)
    board[0, 0] = 1
    board[1, 1] = -1
    board[2, 2] = 1
    return board


@pytest.fixture
def net() -> model_quixo.Net:
    return model_quixo.Net(
        input_shape=model_quixo.OBS_SHAPE, actions_n=game_quixo.N_ACTIONS
    )


# ----------------------------
# Tests for encode_board_for_nn
# ----------------------------


def test_encode_board_for_nn_shape_assert(sample_board):
    dest = np.zeros((999, 5, 5), dtype=np.float32)
    with pytest.raises(AssertionError):
        model_quixo.encode_board_for_nn(dest, sample_board, player=1)


def test_encode_board_for_nn_planes_sum_to_one(sample_board):
    dest = np.zeros(model_quixo.OBS_SHAPE, dtype=np.float32)

    model_quixo.encode_board_for_nn(dest, sample_board, player=1)

    # Each cell should be exactly one-hot encoded across the 3 planes
    # so sum over planes should be 1 everywhere.
    plane_sum = dest.sum(axis=0)
    assert plane_sum.shape == (game_quixo.SIZE, game_quixo.SIZE)
    assert np.allclose(plane_sum, 1.0)


def test_encode_board_for_nn_correct_mapping(sample_board):
    dest = np.zeros(model_quixo.OBS_SHAPE, dtype=np.float32)

    model_quixo.encode_board_for_nn(dest, sample_board, player=1)

    # NOTE: your function flips the row axis:
    # row_idx = SIZE - rev_row_idx - 1
    # so we check the corresponding flipped row positions.

    size = game_quixo.SIZE

    # board[0,0] = 1  -> should appear in plane 0
    row = size - 0 - 1
    col = 0
    assert dest[0, row, col] == 1.0
    assert dest[1, row, col] == 0.0
    assert dest[2, row, col] == 0.0

    # board[1,1] = -1 -> should appear in plane 1
    row = size - 1 - 1
    col = 1
    assert dest[0, row, col] == 0.0
    assert dest[1, row, col] == 1.0
    assert dest[2, row, col] == 0.0

    # empty square -> should appear in plane 2
    row = size - 3 - 1
    col = 3
    assert dest[0, row, col] == 0.0
    assert dest[1, row, col] == 0.0
    assert dest[2, row, col] == 1.0


# ----------------------------
# Tests for states_to_tensor_batch
# ----------------------------


def test_states_to_tensor_batch_shape(sample_board, empty_board):
    states = [sample_board, empty_board]
    players = [1, -1]

    batch_t = model_quixo.states_to_tensor_batch(states, players, device="cpu")

    assert isinstance(batch_t, torch.Tensor)
    assert batch_t.shape == (2,) + model_quixo.OBS_SHAPE
    assert batch_t.dtype == torch.float32


def test_states_to_tensor_batch_device(sample_board):
    states = [sample_board]
    players = [1]

    batch_t = model_quixo.states_to_tensor_batch(states, players, device="cpu")
    assert batch_t.device.type == "cpu"


# ----------------------------
# Tests for Net
# ----------------------------


def test_net_forward_shapes(net):
    x = torch.zeros((4,) + model_quixo.OBS_SHAPE, dtype=torch.float32)

    policy_logits, value = net(x)

    assert policy_logits.shape == (4, game_quixo.N_ACTIONS)
    assert value.shape == (4, 1)


def test_net_value_range(net):
    x = torch.randn((2,) + model_quixo.OBS_SHAPE, dtype=torch.float32)

    _, value = net(x)

    # because you use Tanh() at the end of the value head
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


# ----------------------------
# Fake MCTS for play_game tests
# ----------------------------


class DummyMCTS:
    def __init__(self):
        self.run_calls = 0

    def run_mcts(
        self, n_iterations, n_simulations, state_int, player, net, device="cpu"
    ):
        self.run_calls += 1

    def get_policy_value(self, state_int, tau=1):
        # uniform distribution over actions
        probs = np.ones(game_quixo.N_ACTIONS, dtype=np.float32)
        probs /= probs.sum()
        values = np.zeros(game_quixo.N_ACTIONS, dtype=np.float32)
        return probs, values


# ----------------------------
# Tests for play_game
# ----------------------------


def test_play_game_returns_valid_result(net):
    mcts = [DummyMCTS(), DummyMCTS()]
    replay = collections.deque(maxlen=100)

    result, steps = model_quixo.play_game(
        mcts_stores=mcts,
        replay_buffer=replay,
        net1=net,
        net2=net,
        steps_before_tau_0=2,
        n_iterations=1,
        n_simulations=1,
        net1_plays_first=True,
        device="cpu",
    )

    assert result in (-1, 0, 1)
    assert isinstance(steps, int)
    assert steps >= 0


def test_play_game_populates_replay_buffer(net):
    mcts = [DummyMCTS(), DummyMCTS()]
    replay = collections.deque(maxlen=1000)

    result, steps = model_quixo.play_game(
        mcts_stores=mcts,
        replay_buffer=replay,
        net1=net,
        net2=net,
        steps_before_tau_0=2,
        n_iterations=1,
        n_simulations=1,
        net1_plays_first=True,
        device="cpu",
    )

    # should store at least one position
    assert len(replay) > 0

    state_int, cur_player, probs, value = replay[-1]

    assert isinstance(state_int, int)
    assert cur_player in (1, -1)
    assert len(probs) == game_quixo.N_ACTIONS
    assert value in (-1, 0, 1)


def test_play_game_calls_mcts(net):
    mcts1 = DummyMCTS()
    mcts2 = DummyMCTS()
    mcts = [mcts1, mcts2]

    result, steps = model_quixo.play_game(
        mcts_stores=mcts,
        replay_buffer=None,
        net1=net,
        net2=net,
        steps_before_tau_0=2,
        n_iterations=1,
        n_simulations=1,
        net1_plays_first=True,
        device="cpu",
    )

    assert mcts1.run_calls + mcts2.run_calls > 0
