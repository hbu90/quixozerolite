import collections
import numpy as np

import torch
import torch.nn as nn

from lib import game_quixo, mcts_quixo


OBS_SHAPE = (2, game_quixo.SIZE, game_quixo.SIZE)
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


def _encode_list_state(dest_np, state_list, who_move):
    """ """
    assert dest_np.shape == OBS_SHAPE

    for col_idx, col in enumerate(state_list):
        for rev_row_idx, cell in enumerate(col):
            row_idx = game_quixo.SIZE - rev_row_idx - 1
            if cell == who_move:
                dest_np[0, row_idx, col_idx] = 1.0
            else:
                dest_np[1, row_idx, col_idx] = 1.0


def state_lists_to_batch(state_lists, who_moves_lists, device="cpu"):
    """ """
    assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size,) + OBS_SHAPE, dtype=np.float32)
    for idx, (state, who_move) in enumerate(zip(state_lists, who_moves_lists)):
        _encode_list_state(batch[idx], state, who_move)
    return torch.tensor(batch).to(device)


def play_game(
    mcts_stores,
    replay_buffer,
    net1,
    net2,
    steps_before_tau_0,
    mcts_searches,
    mcts_batch_size,
    net1_plays_first=None,
    device="cpu",
):
    """ """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mcts_quixo.MCTS, type(None), list))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

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
        mcts_stores[cur_player_idx].search_batch(
            mcts_searches,
            mcts_batch_size,
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
