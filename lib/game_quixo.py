import numpy as np
from typing import List
from numpy.typing import NDArray

SIZE = 5
PLAYER_X = 1
PLAYER_O = -1

BORDER_PIECES = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 0),
    (1, 4),
    (2, 0),
    (2, 4),
    (3, 0),
    (3, 4),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
]

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

INITIAL_STATE = np.asarray(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)

POW3_25 = 3 ** np.arange(24, -1, -1, dtype=np.int64)


def legal_directions(row: int, col: int, size: int = 5) -> List[str]:
    """
    Create a list of legal move directions for a Quixo border pieces.

    Args:
        row (): Row index of the border piece.
        col (): Column index of the border piece.
        size (): Size of the board (n x n). Defaults to 5, as per standard Quixo board.
    Returns:
        list: List of valid directions that can be taken.
    """
    dirs = []

    if row != 0:
        dirs.append("DOWN")
    if row != size - 1:
        dirs.append("UP")
    if col != 0:
        dirs.append("RIGHT")
    if col != size - 1:
        dirs.append("LEFT")

    return dirs


# Create list of all possible actions for border pieces - 44 in total
ACTION_MAP = []
for border_index, (border_row, border_col) in enumerate(BORDER_PIECES):
    for legal_dir in legal_directions(border_row, border_col):
        ACTION_MAP.append((border_index, legal_dir))

N_ACTIONS = len(ACTION_MAP)


def encode_board(board: NDArray[np.int8]) -> int:
    """
    Encodes a 5x5 Quixo board into a unique integer.

    Args:
        board (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
    Returns:
        int: Unique integer representing a Quixo board state.
    """
    assert board.shape == (5, 5)

    ternary = (board.astype(np.int64) + 1).ravel()

    return int(ternary @ POW3_25)


def decode_board(value: int) -> NDArray[np.int8]:
    """
    Decodes an integer back into a 5x5 Quixo board.

    Args:
        value (int): Encoded integer that represents a unique Quixo 5x5 board state.
    Returns:
        NDArray[np.int8]: Array that represents a 5x5 Quixo board.
    """
    cells = []
    for _ in range(25):
        value, r = divmod(value, 3)
        cells.append(r - 1)

    cells.reverse()
    return np.array(cells, dtype=np.int8).reshape(5, 5)


def check_won(state: NDArray[np.int8]) -> int:
    """
    Checks if any of players have 5 in a row of their symbol, either horizontally, diagonally, or vertically.

    Args:
        state (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
    Returns:
        int: Integer representing either a winner (1 or -1 for different players) or 0 for no win found.
    """
    for r in range(SIZE):
        row_sum = sum(state[r])
        if row_sum == SIZE:
            return 1
        if row_sum == -SIZE:
            return -1

    for c in range(SIZE):
        col_sum = sum(state[r][c] for r in range(SIZE))
        if col_sum == SIZE:
            return 1
        if col_sum == -SIZE:
            return -1

    diagonal_sum = sum(state[i][i] for i in range(SIZE))
    if diagonal_sum == SIZE:
        return 1
    if diagonal_sum == -SIZE:
        return -1

    anti_diagonal_sum = sum(state[i][SIZE - 1 - i] for i in range(SIZE))
    if anti_diagonal_sum == SIZE:
        return 1
    if anti_diagonal_sum == -SIZE:
        return -1

    return 0


def check_valid(state: NDArray[np.int8], action_idx: int, player: int) -> bool:
    """
    Checks if an action is valid for a player in a given board state.

    Args:
        state (NDArray[np.int8]): Array that represents a 5x5 Quixo board.
        action_idx (int): Index of the action.
        player (int): Integer representing the player (1 or -1).
    Returns:
        bool: Boolean whether the given action is valid.
    """
    assert player in [PLAYER_O, PLAYER_X]

    border_idx, direction = ACTION_MAP[action_idx]
    row, col = BORDER_PIECES[border_idx]

    if state[row][col] not in (0, player):
        return False

    if direction not in legal_directions(row, col):
        return False

    return True


def move(state_int: int, action_idx: int, player: int) -> tuple[int, int]:
    """
    Make a move on the 5x5 Quixo board and output the new board state and whether either player has won.

    Args:
        state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
        action_idx (int): Index of the action.
        player (int): Integer representing the player (1 or -1).
    Returns:
        int: Encoded integer that represents a unique Quixo 5x5 board state.
        int: Indicator whether either player has won (-1 or 1) or no winner with current state (0).
    """
    assert player in [PLAYER_O, PLAYER_X]

    state = decode_board(state_int)
    if check_valid(state, action_idx, player):
        border_idx, direction = ACTION_MAP[action_idx]
        row, col = BORDER_PIECES[border_idx]

        state_new = state.copy()
        if direction == "LEFT":
            for i in range(col, SIZE - 1):
                state_new[row][i] = state_new[row][i + 1]
            state_new[row][SIZE - 1] = player

        elif direction == "RIGHT":
            for i in range(col, 0, -1):
                state_new[row][i] = state_new[row][i - 1]
            state_new[row][0] = player

        elif direction == "UP":
            for i in range(row, SIZE - 1):
                state_new[i][col] = state_new[i + 1][col]
            state_new[SIZE - 1][col] = player

        elif direction == "DOWN":
            for i in range(row, 0, -1):
                state_new[i][col] = state_new[i - 1][col]
            state_new[0][col] = player

        won = check_won(state_new)
        state_new_int = encode_board(state_new)

        return state_new_int, won

    else:
        raise ValueError()


def possible_moves(state_int: int, player: int) -> list:
    """
    Output all possible moves for a player in a given board state.

    Args:
        state_int (int): Encoded integer that represents a unique Quixo 5x5 board state.
        player (int): Integer representing the player (1 or -1).
    Returns:
        list: List of action indices that are valid for the given player.
    """
    assert player in [PLAYER_O, PLAYER_X]

    state = decode_board(state_int)
    return [
        idx for idx, action in enumerate(ACTION_MAP) if check_valid(state, idx, player)
    ]
