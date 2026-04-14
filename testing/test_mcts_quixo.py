import pytest
import numpy as np
import torch
import lib.mcts_quixo as m


class DummyGame:
    N_ACTIONS = 44
    PLAYER_X = 1
    PLAYER_O = -1

    @staticmethod
    def decode_board(state_int):
        return np.zeros((5, 5), dtype=int)

    def possible_moves(self, state_int, player):
        return list(range(44))

    @staticmethod
    def move(state_int, action, player):
        return state_int + 1, 0


class DummyNet(torch.nn.Module):
    @staticmethod
    def forward(batch_v):
        batch_size = batch_v.shape[0]
        logits = torch.zeros((batch_size, 44), dtype=torch.float32)
        values = torch.zeros((batch_size, 1), dtype=torch.float32)
        return logits, values


@pytest.fixture
def mcts():
    return m.MCTS(c_puct=1.0)


@pytest.fixture
def monkey_game(monkeypatch):
    game = DummyGame()
    import lib.mcts_quixo as mm

    monkeypatch.setattr(mm, "game_quixo", game)
    return game


@pytest.fixture
def net():
    return DummyNet()


def test_init_and_clear(mcts):
    mcts.visit_count[0] = [1] * 44
    mcts.clear()
    assert len(mcts.visit_count) == 0


def test_len(mcts):
    mcts.value[0] = [0]
    assert len(mcts) == 1


def test_is_leaf(mcts):
    assert mcts.is_leaf(0)
    mcts.probs[0] = [0] * 44
    assert not mcts.is_leaf(0)


def test_find_leaf_root_noise(mcts, monkey_game):
    state = 0
    player = 1

    mcts.probs[state] = [1 / 44] * 44
    mcts.value_avg[state] = [0.0] * 44
    mcts.visit_count[state] = [1] * 44

    value, s, p, states, actions = mcts.find_leaf(state, player)

    assert value is None
    assert isinstance(states, list)
    assert isinstance(actions, list)


def test_terminal_win(mcts, monkeypatch):
    import lib.mcts_quixo as mm

    def win_move(state, action, player):
        return state + 1, player

    monkeypatch.setattr(mm.game_quixo, "move", win_move)

    mcts.probs[0] = [1 / 44] * 44
    mcts.value_avg[0] = [0.0] * 44
    mcts.visit_count[0] = [1] * 44

    value, *_ = mcts.find_leaf(0, 1)
    assert value == 1.0


def test_terminal_loss(mcts, monkeypatch):
    import lib.mcts_quixo as mm

    def lose_move(state, action, player):
        return state + 1, -player

    monkeypatch.setattr(mm.game_quixo, "move", lose_move)

    mcts.probs[0] = [1 / 44] * 44
    mcts.value_avg[0] = [0.0] * 44
    mcts.visit_count[0] = [1] * 44

    value, *_ = mcts.find_leaf(0, 1)
    assert value == -1.0


def test_no_moves_draw(mcts, monkeypatch):
    import lib.mcts_quixo as mm

    class NoMovesGame(DummyGame):
        def possible_moves(self, state_int, player):
            return []

    monkeypatch.setattr(mm, "game_quixo", NoMovesGame())

    mcts.probs[0] = [1 / 44] * 44
    mcts.value_avg[0] = [0.0] * 44
    mcts.visit_count[0] = [1] * 44

    value, *_ = mcts.find_leaf(0, 1)
    assert value == 0.0


def test_policy_value(mcts):
    state = 0
    mcts.visit_count[state] = [1] * 44
    mcts.value_avg[state] = [0.5] * 44

    probs, values = mcts.get_policy_value(state, tau=1)
    assert abs(sum(probs) - 1.0) < 1e-6

    probs2, _ = mcts.get_policy_value(state, tau=0)
    assert max(probs2) == 1.0


def test_full_mcts_batch(mcts, net, monkeypatch):
    import lib.mcts_quixo as mm

    monkeypatch.setattr(mm, "game_quixo", DummyGame())

    call = {"n": 0}

    def fake_find_leaf(*args):
        if call["n"] == 0:
            call["n"] += 1
            return None, 0, 1, [], []
        return 0.0, 0, 1, [], []

    mcts.find_leaf = fake_find_leaf

    mcts.mcts_simulations_batch(
        n_simulations=2,
        state_int=0,
        player=1,
        net=net,
        device=torch.device("cpu"),
    )

    assert True


def test_run_mcts(mcts, net, monkeypatch):
    import lib.mcts_quixo as mm

    monkeypatch.setattr(mm, "game_quixo", DummyGame())

    mcts.find_leaf = lambda *args: (0.0, 0, 1, [], [])

    mcts.run_mcts(
        n_iterations=1,
        n_simulations=1,
        state_int=0,
        player=1,
        net=net,
        device=torch.device("cpu"),
    )

    assert True


def test_dirichlet_root_noise_branch(mcts, monkeypatch):
    import lib.mcts_quixo as mm

    monkeypatch.setattr(mm, "game_quixo", DummyGame())

    # force deterministic noise
    np.random.seed(0)

    state = 0
    mcts.probs[state] = [1 / 44] * 44
    mcts.value_avg[state] = [0.0] * 44
    mcts.visit_count[state] = [1] * 44

    mcts.find_leaf(state, 1)

    assert True


def test_cycle_draw_branch(mcts, monkeypatch):
    import lib.mcts_quixo as mm

    monkeypatch.setattr(mm, "game_quixo", DummyGame())

    def cycle_move(state, action, player):
        return state, 0

    monkeypatch.setattr(mm.game_quixo, "move", cycle_move)

    mcts.probs[0] = [1 / 44] * 44
    mcts.value_avg[0] = [0.0] * 44
    mcts.visit_count[0] = [1] * 44

    value, *_ = mcts.find_leaf(0, 1)

    assert value == 0.0


def test_expand_queue_branch(mcts, net, monkeypatch):
    import lib.mcts_quixo as mm

    monkeypatch.setattr(mm, "game_quixo", DummyGame())

    def fake_find_leaf(*args):
        return None, 0, 1, [], []

    mcts.find_leaf = fake_find_leaf

    mcts.mcts_simulations_batch(
        n_simulations=1,
        state_int=0,
        player=1,
        net=net,
        device=torch.device("cpu"),
    )

    assert True


def test_prob_mask_zero_branch(mcts, net, monkeypatch):
    import lib.mcts_quixo as mm

    class BadGame(DummyGame):
        def possible_moves(self, state_int, player):
            return []

    monkeypatch.setattr(mm, "game_quixo", BadGame())

    def fake_find_leaf(*args):
        return None, 0, 1, [], []

    mcts.find_leaf = fake_find_leaf

    mcts.mcts_simulations_batch(1, 0, 1, net, torch.device("cpu"))

    assert True
