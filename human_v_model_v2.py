import pygame
import sys
import numpy as np
from lib import game, mcts, model
import torch

# ----------------- CONFIG -----------------
CELL_SIZE = 100
ROWS = game.GAME_ROWS
COLS = game.GAME_COLS
WIDTH = COLS * CELL_SIZE
HEIGHT = (ROWS+1) * CELL_SIZE  # Extra row for input
RADIUS = CELL_SIZE // 2 - 5
FPS = 30
MCTS_SEARCHES = 20
MCTS_BATCH_SIZE = 4

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
YELLOW = (255, 215, 0)
BLUE = (0, 100, 255)

# Player encoding
HUMAN = 0
AI = 1


# ----------------- UTILS -----------------
def draw_board(screen, board, last_move=None):
    screen.fill(BLUE)
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c*CELL_SIZE, (r+1)*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.circle(screen, BLACK, (c*CELL_SIZE + RADIUS, (r+1)*CELL_SIZE + RADIUS), RADIUS)
    for c, col in enumerate(board):
        for r, cell in enumerate(col):
            color = RED if cell == 1 else YELLOW if cell == 0 else BLACK
            pygame.draw.circle(screen, color, (c*CELL_SIZE + RADIUS, HEIGHT - (r+1)*CELL_SIZE + RADIUS), RADIUS)
    if last_move:
        lr, lc = last_move
        pygame.draw.circle(screen, WHITE, (lc*CELL_SIZE + RADIUS, HEIGHT - (lr+1)*CELL_SIZE + RADIUS), RADIUS//2)
    pygame.display.update()


def get_human_move(state_list):
    legal_cols = [c for c, col in enumerate(state_list) if len(col) < ROWS]
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, _ = event.pos
                col = x // CELL_SIZE
                if col in legal_cols:
                    return col


def ai_move(state_int, net, device):
    mcts_store = mcts.MCTS()
    mcts_store.search_batch(MCTS_SEARCHES, MCTS_BATCH_SIZE, state_int, player=AI, net=net, device=device)
    probs, _ = mcts_store.get_policy_value(state_int, tau=0)
    action = int(np.argmax(probs))
    new_state, won = game.move(state_int, action, AI)
    last_move = (len(game.decode_binary(new_state)[action])-1, action)
    return new_state, last_move, won


# ----------------- MAIN -----------------
def main(model_path, human_first=True, use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    net = model.Net(model.OBS_SHAPE, game.GAME_COLS)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4: Human vs AI")
    clock = pygame.time.Clock()

    state = game.encode_lists([[]]*COLS)
    human_player = HUMAN if human_first else AI
    current_player = 0  # always start with player 0
    last_move = None
    running = True
    winner = None

    while running:
        state_list = game.decode_binary(state)
        draw_board(screen, state_list, last_move)

        if current_player == human_player:
            col = get_human_move(state_list)
            state, won = game.move(state, col, human_player)
            last_move = (len(game.decode_binary(state)[col])-1, col)
            if won:
                winner = "You"
                running = False
        else:
            pygame.display.set_caption("AI is thinking...")
            pygame.display.update()
            pygame.time.delay(500)
            state, last_move, won = ai_move(state, net, device)
            if won:
                winner = "AI"
                running = False

        current_player = 1 - current_player  # alternate turn

        if not game.possible_moves(state):
            winner = "Draw"
            running = False

        clock.tick(FPS)

    draw_board(screen, game.decode_binary(state), last_move)
    pygame.time.delay(1000)
    if winner == "Draw":
        print("Game ended in a draw!")
    else:
        print(f"{winner} won the game!")

    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained model")
    parser.add_argument("--human-first", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args.model, args.human_first, args.cuda)
