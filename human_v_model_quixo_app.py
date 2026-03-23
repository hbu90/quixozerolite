import pygame
import sys
import numpy as np
import torch
from tkinter import filedialog, Tk

from lib import game_quixo, model_quixo

pygame.init()

WIDTH = 700
HEIGHT = 750
CELL = 100
BOARD_SIZE = 5
MARGIN = 100

WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GREEN = (120, 220, 120)
RED = (220, 120, 120)
GRAY = (200, 200, 200)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quixo RL")

font = pygame.font.SysFont(None, 48)
small_font = pygame.font.SysFont(None, 30)


class QuixoApp:

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = game_quixo.encode_board(game_quixo.INITIAL_STATE)
        self.player = 1
        self.human_player = 1
        self.net = None
        self.selected = None
        self.cube_actions = []
        self.game_over = False
        self.message = "Load model (L), choose player (1=you, 2=agent)"

    def load_model(self):
        Tk().withdraw()

        path = filedialog.askopenfilename(filetypes=[("Model", "*.dat")])

        if not path:
            return

        net = model_quixo.Net(model_quixo.OBS_SHAPE, game_quixo.N_ACTIONS)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()

        self.net = net
        self.message = "Model loaded!"

    def draw(self):
        screen.fill(WHITE)

        board = game_quixo.decode_board(self.state)

        # draw board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):

                x = MARGIN + c * CELL
                y = MARGIN + r * CELL

                rect = pygame.Rect(x, y, CELL, CELL)

                color = GRAY if (r, c) == self.selected else BLACK
                pygame.draw.rect(screen, color, rect, 2)

                val = board[r][c]

                if val != 0:
                    text = "X" if val == 1 else "O"
                    label = font.render(text, True, BLACK)
                    screen.blit(label, (x + 35, y + 30))

        # highlight legal cubes (only on your turn)
        if not self.game_over and self.player == self.human_player:
            legal = game_quixo.possible_moves(self.state, self.player)
            legal_cubes = set()

            for a in legal:
                border_idx, _ = game_quixo.ACTION_MAP[a]
                r, c = game_quixo.BORDER_SQUARES[border_idx]
                legal_cubes.add((r, c))

            for r, c in legal_cubes:
                x = MARGIN + c * CELL
                y = MARGIN + r * CELL
                pygame.draw.rect(screen, GREEN, (x, y, CELL, CELL), 4)

        # message
        msg = small_font.render(self.message, True, BLACK)
        screen.blit(msg, (20, 20))

        # TURN INDICATOR
        if not self.game_over:
            symbol = "X" if self.player == 1 else "O"

            if self.player == self.human_player:
                turn_text = f"Your turn ({symbol})"
                color = GREEN
            else:
                turn_text = f"Agent thinking ({symbol})"
                color = RED

            turn_label = small_font.render(turn_text, True, color)
            screen.blit(turn_label, (20, 60))

        pygame.display.flip()

    def select_cube(self, r, c):
        if self.game_over or self.player != self.human_player:
            return

        legal = game_quixo.possible_moves(self.state, self.player)

        cube_actions = [
            a for a in legal
            if game_quixo.BORDER_SQUARES[
                   game_quixo.ACTION_MAP[a][0]
               ] == (r, c)
        ]

        if not cube_actions:
            return

        self.selected = (r, c)
        self.cube_actions = cube_actions
        self.message = "Use arrow keys to push"

    def push_direction(self, key):
        if self.selected is None or self.game_over or self.player != self.human_player:
            return

        direction_map = {
            pygame.K_UP: "UP",
            pygame.K_DOWN: "DOWN",
            pygame.K_LEFT: "LEFT",
            pygame.K_RIGHT: "RIGHT"
        }

        if key not in direction_map:
            return

        direction = direction_map[key]

        for action in self.cube_actions:
            _, d = game_quixo.ACTION_MAP[action]

            if d == direction:
                self.apply_move(action)
                return

    def apply_move(self, action):
        if self.game_over:
            return

        self.state, won = game_quixo.move(self.state, action, self.player)

        if won:
            self.message = "You win!"
            self.game_over = True
            return

        self.player *= -1
        self.selected = None

        pygame.time.wait(150)
        self.agent_move()

    def agent_move(self):
        if self.net is None or self.game_over:
            return

        self.message = "Agent thinking..."
        pygame.display.flip()

        board_np = game_quixo.decode_board(self.state)

        # FAST encoding (no helper function)
        current = (board_np == self.player).astype(np.float32)
        opponent = (board_np == -self.player).astype(np.float32)

        batch = torch.tensor(
            np.stack([current, opponent]),
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.inference_mode():
            logits, _ = self.net(batch)
            probs = torch.softmax(logits, dim=1)[0]

        legal = game_quixo.possible_moves(self.state, self.player)

        if not legal:
            self.message = "No legal moves!"
            return

        mask = torch.zeros(game_quixo.N_ACTIONS)
        mask[legal] = 1

        probs = probs * mask

        if probs.sum() == 0:
            probs[legal] = 1.0 / len(legal)
        else:
            probs = probs / probs.sum()

        action = torch.multinomial(probs, 1).item()

        self.state, won = game_quixo.move(self.state, action, self.player)

        if won:
            self.message = "Agent wins!"
            self.game_over = True
            return

        self.player *= -1
        self.message = "Your turn"

    def click(self, pos):
        if self.game_over or self.player != self.human_player:
            return

        x, y = pos

        c = (x - MARGIN) // CELL
        r = (y - MARGIN) // CELL

        if r < 0 or r >= 5 or c < 0 or c >= 5:
            return

        self.select_cube(r, c)


app = QuixoApp()
clock = pygame.time.Clock()

while True:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            app.click(pygame.mouse.get_pos())

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_l:
                app.load_model()

            if event.key == pygame.K_1:
                app.player = 1
                app.human_player = 1
                app.message = "You start (X)"

            if event.key == pygame.K_2:
                app.player = -1
                app.human_player = 1
                app.message = "Agent starts (O)"
                app.agent_move()

            if event.key == pygame.K_r:
                app.reset()

            app.push_direction(event.key)

    app.draw()
    clock.tick(60)
