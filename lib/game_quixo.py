import numpy as np

SIZE = 5
PLAYER_X = 1
PLAYER_O = -1

BORDER_SQUARES = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 4),
    (2, 0), (2, 4),
    (3, 0), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
]

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

INITIAL_STATE = [
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
]


def legal_directions(row, col, size=5):
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


ACTION_MAP = []
for idx, (row, col) in enumerate(BORDER_SQUARES):
    for dir in legal_directions(row, col):
        ACTION_MAP.append((idx, dir))

N_ACTIONS = len(ACTION_MAP)


def encode_board(board):
    """
    Encodes a 5x5 Quixo board into a unique integer.
    board: array-like shape (5,5) with values {-1, 0, 1}
    """
    board = np.asarray(board, dtype=np.int8)
    assert board.shape == (5, 5)

    # Map {-1,0,1} → {0,1,2}
    trits = board + 1

    value = 0
    for t in trits.flatten():
        value = value * 3 + int(t)

    return value


def decode_board(value):
    """
    Decodes an integer back into a 5x5 Quixo board.
    """
    cells = []
    for _ in range(25):
        value, r = divmod(value, 3)
        cells.append(r - 1)

    cells.reverse()
    return np.array(cells, dtype=np.int8).reshape(5, 5)


def check_won(state):
    """
    Checks if any of players have 5 in a row of their symbol, either horizontally, diagonally, or vertically.
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


def check_valid(state, player, action_idx):
    border_idx, direction = ACTION_MAP[action_idx]
    row, col = BORDER_SQUARES[border_idx]

    if state[row][col] not in (0, player):
        return False

    if direction not in legal_directions(row, col):
        return False

    return True


def move(state_int, action_idx, player):
    """
    """
    state = decode_board(state_int)
    if check_valid(state, player, action_idx):
        border_idx, direction = ACTION_MAP[action_idx]
        row, col = BORDER_SQUARES[border_idx]

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


def possible_moves(state_int, player):
    """
    """
    state = decode_board(state_int)
    return [
        idx
        for idx, action in enumerate(ACTION_MAP)
        if check_valid(state, player, idx)
    ]
