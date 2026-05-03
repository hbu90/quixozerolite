import pygame
import sys
import numpy as np
import torch
from tkinter import filedialog, Tk

from lib import game_quixo, model_quixo

pygame.init()

WIDTH = 900
HEIGHT = 780
CELL = 100
BOARD_SIZE = 5
MARGIN_X = 80
MARGIN_Y = 140

WHITE = (245, 245, 245)
BLACK = (25, 25, 25)
GRAY = (200, 200, 200)
DARK_GRAY = (120, 120, 120)

GREEN = (120, 220, 120)
RED = (220, 120, 120)
BLUE = (120, 160, 240)
YELLOW = (240, 220, 120)

WOOD = (220, 190, 140)
WOOD_DARK = (180, 140, 90)
WOOD_LIGHT = (245, 225, 190)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quixo RL")

font = pygame.font.SysFont(None, 56)
small_font = pygame.font.SysFont(None, 28)
tiny_font = pygame.font.SysFont(None, 22)


def draw_3d_tile(x, y, size, base_color, highlight=False):
    shadow_offset = 6
    pygame.draw.rect(
        screen,
        (150, 150, 150),
        pygame.Rect(x + shadow_offset, y + shadow_offset, size, size),
        border_radius=8,
    )

    pygame.draw.rect(
        screen,
        base_color,
        pygame.Rect(x, y, size, size),
        border_radius=8,
    )

    pygame.draw.line(screen, WOOD_LIGHT, (x, y), (x + size, y), 3)
    pygame.draw.line(screen, WOOD_LIGHT, (x, y), (x, y + size), 3)

    pygame.draw.line(screen, WOOD_DARK, (x, y + size), (x + size, y + size), 3)
    pygame.draw.line(screen, WOOD_DARK, (x + size, y), (x + size, y + size), 3)

    if highlight:
        pygame.draw.rect(
            screen,
            BLACK,
            pygame.Rect(x - 2, y - 2, size + 4, size + 4),
            4,
            border_radius=10,
        )


class QuixoApp:
    def __init__(self):
        self.mode = "LOAD"
        self.reset()

    def reset(self):
        self.state = game_quixo.encode_board(game_quixo.INITIAL_STATE)
        self.player = 1
        self.human_player = 1
        self.net = None

        self.selected = None
        self.cube_actions = []
        self.available_dirs = []

        self.game_over = False
        self.message = ""

        self.moves_x = []
        self.moves_o = []

    def start_game(self):
        self.reset()
        self.mode = "GAME"
        self.message = "Press L to load model. Press 1 or 2 to choose who starts."

    def load_model(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(filetypes=[("Model", "*.dat")])

        if not path:
            return

        net = model_quixo.Net(
            input_shape=model_quixo.OBS_SHAPE, actions_n=game_quixo.N_ACTIONS
        )

        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()

        self.net = net
        self.message = f"Loaded: {path.split('/')[-1]}"

    def record_move(self, player, action):
        border_idx, direction = game_quixo.ACTION_MAP[action]
        r, c = game_quixo.BORDER_PIECES[border_idx]
        text = f"({r},{c}) {direction}"

        if player == 1:
            self.moves_x.append(text)
        else:
            self.moves_o.append(text)

    def draw_load_screen(self):
        screen.fill(WHITE)

        title = font.render("QUIXO RL", True, BLACK)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 160))

        subtitle = small_font.render("Press SPACE to start", True, DARK_GRAY)
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 240))

        controls = [
            "Controls:",
            "L = Load Model",
            "1 = You start as X",
            "2 = Agent starts as X",
            "R = Reset game",
            "Click border cube, then use Arrow Keys to push",
        ]

        y = 330
        for line in controls:
            label = small_font.render(line, True, BLACK)
            screen.blit(label, (WIDTH // 2 - label.get_width() // 2, y))
            y += 35

        pygame.display.flip()

    def draw_moves_panel(self):
        panel_x = 610
        panel_y = 140

        pygame.draw.rect(
            screen,
            (235, 235, 235),
            pygame.Rect(panel_x, panel_y, 260, 520),
            border_radius=12,
        )
        pygame.draw.rect(
            screen,
            (170, 170, 170),
            pygame.Rect(panel_x, panel_y, 260, 520),
            2,
            border_radius=12,
        )

        title = small_font.render("Move History", True, BLACK)
        screen.blit(title, (panel_x + 60, panel_y + 15))

        x_label = tiny_font.render("X moves:", True, BLACK)
        o_label = tiny_font.render("O moves:", True, BLACK)
        screen.blit(x_label, (panel_x + 20, panel_y + 55))
        screen.blit(o_label, (panel_x + 140, panel_y + 55))

        max_lines = 18
        x_moves = self.moves_x[-max_lines:]
        o_moves = self.moves_o[-max_lines:]

        y = panel_y + 80
        for i in range(max(len(x_moves), len(o_moves))):
            if i < len(x_moves):
                label = tiny_font.render(x_moves[i], True, (40, 40, 40))
                screen.blit(label, (panel_x + 20, y))

            if i < len(o_moves):
                label = tiny_font.render(o_moves[i], True, (40, 40, 40))
                screen.blit(label, (panel_x + 140, y))

            y += 24

    def draw(self):
        if self.mode == "LOAD":
            self.draw_load_screen()
            return

        screen.fill(WHITE)

        board = game_quixo.decode_board(self.state)

        pygame.draw.rect(
            screen,
            WOOD,
            pygame.Rect(MARGIN_X - 20, MARGIN_Y - 20, CELL * 5 + 40, CELL * 5 + 40),
            border_radius=18,
        )

        pygame.draw.rect(
            screen,
            WOOD_DARK,
            pygame.Rect(MARGIN_X - 20, MARGIN_Y - 20, CELL * 5 + 40, CELL * 5 + 40),
            4,
            border_radius=18,
        )

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = MARGIN_X + c * CELL
                y = MARGIN_Y + r * CELL

                base = WOOD_LIGHT
                highlight = (r, c) == self.selected

                draw_3d_tile(x, y, CELL, base, highlight=highlight)

                val = board[r][c]
                if val != 0:
                    text = "X" if val == 1 else "O"
                    color = (50, 80, 200) if val == 1 else (200, 60, 60)
                    label = font.render(text, True, color)
                    screen.blit(label, (x + 34, y + 24))

        if not self.game_over and self.player == self.human_player:
            legal = game_quixo.possible_moves(self.state, self.player)
            legal_cubes = set()

            for a in legal:
                border_idx, _ = game_quixo.ACTION_MAP[a]
                r, c = game_quixo.BORDER_PIECES[border_idx]
                legal_cubes.add((r, c))

            for r, c in legal_cubes:
                x = MARGIN_X + c * CELL
                y = MARGIN_Y + r * CELL
                pygame.draw.rect(
                    screen,
                    GREEN,
                    pygame.Rect(x + 4, y + 4, CELL - 8, CELL - 8),
                    4,
                    border_radius=10,
                )

        msg = small_font.render(self.message, True, BLACK)
        screen.blit(msg, (20, 20))

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

        if self.selected is not None and self.available_dirs:
            dirs_text = "  ".join(self.available_dirs)
            hint = tiny_font.render(f"Available pushes: {dirs_text}", True, BLACK)
            screen.blit(hint, (20, 100))

            hint2 = tiny_font.render("Use Arrow Keys (↑ ↓ ← →)", True, DARK_GRAY)
            screen.blit(hint2, (20, 120))

        self.draw_moves_panel()

        pygame.display.flip()

    def select_cube(self, r, c):
        if self.game_over or self.player != self.human_player:
            return

        legal = game_quixo.possible_moves(self.state, self.player)

        cube_actions = [
            a
            for a in legal
            if game_quixo.BORDER_PIECES[game_quixo.ACTION_MAP[a][0]] == (r, c)
        ]

        if not cube_actions:
            return

        self.selected = (r, c)
        self.cube_actions = cube_actions

        dirs = []
        for action in cube_actions:
            _, d = game_quixo.ACTION_MAP[action]
            dirs.append(d)

        self.available_dirs = sorted(set(dirs))
        self.message = "Cube selected"

    def push_direction(self, key):
        if self.selected is None or self.game_over or self.player != self.human_player:
            return

        direction_map = {
            pygame.K_UP: "UP",
            pygame.K_DOWN: "DOWN",
            pygame.K_LEFT: "LEFT",
            pygame.K_RIGHT: "RIGHT",
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

        self.record_move(self.player, action)

        self.state, won = game_quixo.move(self.state, action, self.player)

        if won:
            if self.player == self.human_player:
                self.message = "You win!"
            else:
                self.message = "Agent wins!"
            self.game_over = True
            return

        self.player *= -1
        self.selected = None
        self.cube_actions = []
        self.available_dirs = []

        pygame.time.wait(1500)

        if self.player != self.human_player:
            self.agent_move()

    def agent_move(self):
        if self.net is None or self.game_over:
            self.message = "No model loaded!"
            return

        self.message = "Agent thinking..."
        pygame.display.flip()

        board_np = game_quixo.decode_board(self.state)

        batch = model_quixo.states_to_tensor_batch(
            [board_np], [self.player], device="cpu"
        )

        with torch.inference_mode():
            logits, _ = self.net(batch)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        legal = game_quixo.possible_moves(self.state, self.player)

        if not legal:
            self.message = "No legal moves!"
            self.game_over = True
            return

        mask = np.zeros(game_quixo.N_ACTIONS, dtype=np.float32)
        mask[legal] = 1.0

        probs = probs * mask

        if probs.sum() == 0:
            probs[legal] = 1.0 / len(legal)
        else:
            probs = probs / probs.sum()

        action = np.random.choice(game_quixo.N_ACTIONS, p=probs)

        self.record_move(self.player, action)

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

        c = (x - MARGIN_X) // CELL
        r = (y - MARGIN_Y) // CELL

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

        if event.type == pygame.KEYDOWN:
            if app.mode == "LOAD":
                if event.key == pygame.K_SPACE:
                    app.start_game()
                continue

            if event.key == pygame.K_l:
                app.load_model()

            if event.key == pygame.K_1:
                app.reset()
                app.mode = "GAME"
                app.human_player = 1
                app.player = 1
                app.message = "You start (X)"

            if event.key == pygame.K_2:
                app.reset()
                app.mode = "GAME"
                app.human_player = -1
                app.player = 1
                app.message = "Agent starts (X)"
                app.agent_move()

            if event.key == pygame.K_r:
                app.reset()
                app.mode = "GAME"
                app.message = "Game reset"

            app.push_direction(event.key)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if app.mode == "GAME":
                app.click(pygame.mouse.get_pos())

    app.draw()
    clock.tick(60)
