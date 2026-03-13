import pygame
import sys

BOARD_SIZE = 5
PLAYER_X = 1
PLAYER_O = 0
EMPTY = -1

CELL_SIZE = 100
BOARD_SIZE = 5
WIDTH = CELL_SIZE * BOARD_SIZE
HEIGHT = CELL_SIZE * BOARD_SIZE
RADIUS = CELL_SIZE // 2 - 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 200)
RED = (220, 50, 50)
YELLOW = (255, 215, 0)


def create_board():
    return [[EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]


def is_edge(row, col):
    return row == 0 or row == BOARD_SIZE-1 or col == 0 or col == BOARD_SIZE-1


def check_win(board, player):
    # Rows
    for r in range(BOARD_SIZE):
        if all(board[r][c] == player for c in range(BOARD_SIZE)):
            return True
    # Columns
    for c in range(BOARD_SIZE):
        if all(board[r][c] == player for r in range(BOARD_SIZE)):
            return True
    # Diagonals
    if all(board[i][i] == player for i in range(BOARD_SIZE)):
        return True
    if all(board[i][BOARD_SIZE-1-i] == player for i in range(BOARD_SIZE)):
        return True
    return False


def get_legal_moves(board, player):
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if is_edge(r, c) and (board[r][c] == EMPTY or board[r][c] == player):
                moves.append((r, c))
    return moves


def make_move(board, row, col, direction, player):
    """
    Push the cube in the given direction (up/down/left/right)
    direction: 'up', 'down', 'left', 'right'
    """
    new_board = [r.copy() for r in board]
    if direction == 'up':
        col_vals = [new_board[r][col] for r in range(BOARD_SIZE)]
        col_vals = [player] + col_vals[:-1]
        for r in range(BOARD_SIZE):
            new_board[r][col] = col_vals[r]
    elif direction == 'down':
        col_vals = [new_board[r][col] for r in range(BOARD_SIZE)]
        col_vals = col_vals[1:] + [player]
        for r in range(BOARD_SIZE):
            new_board[r][col] = col_vals[r]
    elif direction == 'left':
        row_vals = new_board[row][:]
        row_vals = [player] + row_vals[:-1]
        new_board[row] = row_vals
    elif direction == 'right':
        row_vals = new_board[row][:]
        row_vals = row_vals[1:] + [player]
        new_board[row] = row_vals
    return new_board


def draw_board(screen, board):
    screen.fill(BLUE)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            pygame.draw.rect(screen, WHITE, (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            val = board[r][c]
            if val != EMPTY:
                color = RED if val == PLAYER_X else YELLOW
                pygame.draw.circle(screen, color, (c*CELL_SIZE + RADIUS, r*CELL_SIZE + RADIUS), RADIUS)
    pygame.display.update()


def get_cell(pos):
    x, y = pos
    return y // CELL_SIZE, x // CELL_SIZE


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quixo")
    clock = pygame.time.Clock()

    board = create_board()
    current_player = PLAYER_X
    running = True

    while running:
        draw_board(screen, board)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                row, col = get_cell(event.pos)
                legal_moves = get_legal_moves(board, current_player)
                if (row, col) in legal_moves:
                    # For now, pick default direction (right or down) for testing
                    direction = 'right' if row in [0, BOARD_SIZE-1] else 'down'
                    board = make_move(board, row, col, direction, current_player)
                    if check_win(board, current_player):
                        draw_board(screen, board)
                        print("Player", "X" if current_player==PLAYER_X else "O", "wins!")
                        pygame.time.delay(2000)
                        running = False
                    current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

        clock.tick(30)


if __name__ == "__main__":
    main()
