import pytest
import numpy as np
from unittest.mock import MagicMock
from lib.mcts_quixo import MCTS


class DummyNet:
    def __call__(self, batch_v):
        batch_size = batch_v.shape[0]
        logits_v = MagicMock()
        values_v = MagicMock()
        logits_v.data = MagicMock()
        values_v.data = MagicMock()
        logits_v.shape = (batch_size, 44)
        values_v.shape = (batch_size, 1)
        logits_v = np.zeros((batch_size, 44))
        values_v = np.zeros((batch_size, 1))
        return MagicMock(data=MagicMock(cpu=lambda: logits_v)), MagicMock(data=MagicMock(cpu=lambda: values_v))


class DummyGame:
    N_ACTIONS = 44
    def __init__(self):
        self.calls = 0
    def decode_board(self, state_int):
        return np.zeros((5,5), dtype=int)
    def possible_moves(self, state_int, player):
        return list(range(44))
    def move(self, state_int, action_idx, player):
        return state_int + 1, 0


@pytest.fixture
def mcts():
    import lib.mcts_quixo as m
    return MCTS(c_puct=1.0)


@pytest.fixture
def dummy_game(monkeypatch):
    game = DummyGame()
    import lib.mcts_quixo as m
    monkeypatch.setattr(m, "game_quixo", game)
    return game


@pytest.fixture
def dummy_net():
    return DummyNet()


def test_mcts_initialization(mcts):
    assert isinstance(mcts.visit_count, dict)
    assert len(mcts) == 0


def test_is_leaf(mcts):
    assert mcts.is_leaf(0) is True
    mcts.probs[0] = [0]*44
    assert mcts.is_leaf(0) is False


def test_find_leaf_returns_values(mcts, dummy_game):
    state_int = 0
    player = 0
    mcts.probs[state_int] = [1/44]*44
    mcts.value_avg[state_int] = [0]*44
    mcts.visit_count[state_int] = [0]*44
    value, leaf_state, leaf_player, states, actions = mcts.find_leaf(state_int, player)
    assert leaf_state == state_int
    assert leaf_player in (0,1)
    assert isinstance(states, list)
    assert isinstance(actions, list)


def test_search_minibatch_runs(mcts, dummy_game, dummy_net):
    state_int = 0
    player = 0
    mcts.probs[state_int] = [1/44]*44
    mcts.value_avg[state_int] = [0]*44
    mcts.visit_count[state_int] = [0]*44
    mcts.search_minibatch(1, state_int, player, dummy_net)
    assert len(mcts.visit_count) > 0


def test_get_policy_value(mcts, dummy_game):
    state_int = 0
    mcts.visit_count[state_int] = [1]*44
    mcts.value_avg[state_int] = [0.5]*44
    probs, values = mcts.get_policy_value(state_int)
    assert np.isclose(sum(probs), 1.0)
    assert len(values) == 44


def test_clear(mcts, dummy_game):
    state_int = 0
    mcts.visit_count[state_int] = [1]*44
    mcts.clear()
    assert len(mcts.visit_count) == 0
