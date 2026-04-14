import pytest
import numpy as np
from lib.game_quixo import (
    encode_board,
    decode_board,
    check_won,
    check_valid,
    move,
    possible_moves,
    INITIAL_STATE,
    N_ACTIONS,
    ACTION_MAP,
    PLAYER_X,
    PLAYER_O,
    BORDER_PIECES,
)


def test_encode_decode_identity():
    board = np.array(INITIAL_STATE)
    val = encode_board(board)
    board2 = decode_board(val)
    np.testing.assert_array_equal(board, board2)


def test_check_won_empty():
    board = np.array(INITIAL_STATE)
    assert check_won(board) == 0


def test_check_won_horizontal():
    board = np.array(INITIAL_STATE)
    board[0] = [1, 1, 1, 1, 1]
    assert check_won(board) == 1
    board[0] = [-1, -1, -1, -1, -1]
    assert check_won(board) == -1


def test_check_won_vertical():
    board = np.array(INITIAL_STATE)
    for r in range(5):
        board[r][0] = 1
    assert check_won(board) == 1
    board = np.array(INITIAL_STATE)
    for r in range(5):
        board[r][1] = -1
    assert check_won(board) == -1


def test_check_won_diagonal():
    board = np.array(INITIAL_STATE)
    for i in range(5):
        board[i][i] = 1
    assert check_won(board) == 1
    board = np.array(INITIAL_STATE)
    for i in range(5):
        board[i][4 - i] = 1
    assert check_won(board) == 1
    board = np.array(INITIAL_STATE)
    for i in range(5):
        board[i][i] = -1
    assert check_won(board) == -1
    board = np.array(INITIAL_STATE)
    for i in range(5):
        board[i][4 - i] = -1
    assert check_won(board) == -1


def test_check_valid_and_possible_moves():
    board_int = encode_board(INITIAL_STATE)
    moves = possible_moves(board_int, PLAYER_X)
    board = decode_board(board_int)
    assert all(check_valid(board, m, PLAYER_X) for m in moves)
    assert len(moves) > 0


def test_check_valid_rejects_opponent_piece():
    state = np.zeros((5, 5), dtype=np.int8)
    action_idx = 0
    border_idx, _ = ACTION_MAP[action_idx]
    row, col = BORDER_PIECES[border_idx]
    state[row][col] = PLAYER_O

    assert check_valid(state, action_idx, PLAYER_X) is False


def test_check_valid_false_due_to_legal_directions(monkeypatch):
    state = np.array(INITIAL_STATE)
    action_idx = 0
    border_idx, direction = ACTION_MAP[action_idx]
    row, col = BORDER_PIECES[border_idx]
    state[row][col] = PLAYER_X
    monkeypatch.setattr("lib.game_quixo.legal_directions", lambda r, c: [])

    assert check_valid(state, action_idx, PLAYER_X) is False


def test_move_changes_state():
    board_int = encode_board(INITIAL_STATE)
    action_idx = possible_moves(board_int, PLAYER_X)[0]
    new_int, won = move(board_int, action_idx, PLAYER_X)
    assert isinstance(new_int, int)
    assert won in [-1, 0, 1]
    assert new_int != board_int


def test_move_invalid_raises():
    board = np.array(INITIAL_STATE)
    invalid_idx = len(ACTION_MAP)
    with pytest.raises(IndexError):
        check_valid(board, invalid_idx, PLAYER_X)


def test_move_raises_value_error():
    state = np.asarray(
        [
            [1, 1, 0, -1, -1],
            [1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    state_int = encode_board(state)

    player = PLAYER_X

    valid_actions = set(possible_moves(state_int, player))
    all_actions = set(range(N_ACTIONS))
    invalid_actions = list(all_actions - valid_actions)
    bad_action = invalid_actions[0]

    with pytest.raises(ValueError):
        move(state_int, bad_action, player)
