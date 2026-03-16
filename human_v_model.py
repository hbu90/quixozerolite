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

WHITE = (245,245,245)
BLACK = (30,30,30)
BLUE = (120,170,255)
GREEN = (120,220,120)
RED = (220,120,120)
GRAY = (200,200,200)

screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Quixo RL")

font = pygame.font.SysFont(None,48)
small_font = pygame.font.SysFont(None,30)


class QuixoApp:

    def __init__(self):

        self.state = game_quixo.encode_board(game_quixo.INITIAL_STATE)
        self.player = 1
        self.net = None

        self.selected = None
        self.cube_actions = []

        self.message = "Load model (L) and choose first player (1 or 2)"

    # --------------------------
    # MODEL LOADING
    # --------------------------

    def load_model(self):

        Tk().withdraw()

        path = filedialog.askopenfilename(filetypes=[("Model","*.dat")])

        if not path:
            return

        net = model_quixo.Net(model_quixo.OBS_SHAPE, game_quixo.N_ACTIONS)
        net.load_state_dict(torch.load(path,map_location="cpu"))
        net.eval()

        self.net = net

        self.message = "Model loaded!"

    # --------------------------
    # DRAW BOARD
    # --------------------------

    def draw(self):

        screen.fill(WHITE)

        board = game_quixo.decode_board(self.state)

        # draw board cells
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):

                x = MARGIN + c*CELL
                y = MARGIN + r*CELL

                rect = pygame.Rect(x,y,CELL,CELL)

                color = GRAY if (r,c)==self.selected else BLACK

                pygame.draw.rect(screen,color,rect,2)

                val = board[r][c]

                if val != 0:

                    text = "X" if val==1 else "O"

                    label = font.render(text,True,BLACK)

                    screen.blit(label,(x+35,y+30))

        # highlight legal cubes
        legal = game_quixo.possible_moves(self.state,self.player)

        legal_cubes = set()

        for a in legal:
            r,c,_ = game_quixo.decode_action(a)
            legal_cubes.add((r,c))

        for r,c in legal_cubes:

            x = MARGIN + c*CELL
            y = MARGIN + r*CELL

            pygame.draw.rect(screen,GREEN,(x,y,CELL,CELL),4)

        # draw message
        msg = small_font.render(self.message,True,BLACK)
        screen.blit(msg,(20,20))

        pygame.display.flip()

    # --------------------------
    # HUMAN MOVE
    # --------------------------

    def select_cube(self,r,c):

        legal = game_quixo.possible_moves(self.state,self.player)

        cube_actions = [
            a for a in legal
            if game_quixo.decode_action(a)[0:2]==(r,c)
        ]

        if not cube_actions:
            return

        self.selected = (r,c)
        self.cube_actions = cube_actions

        self.message = "Use arrow keys to push"

    # --------------------------
    # PUSH DIRECTION
    # --------------------------

    def push_direction(self,key):

        if self.selected is None:
            return

        direction_map = {
            pygame.K_UP:"UP",
            pygame.K_DOWN:"DOWN",
            pygame.K_LEFT:"LEFT",
            pygame.K_RIGHT:"RIGHT"
        }

        if key not in direction_map:
            return

        direction = direction_map[key]

        for action in self.cube_actions:

            r,c,d = game_quixo.decode_action(action)

            if d == direction:

                self.apply_move(action)
                return

    # --------------------------
    # APPLY MOVE
    # --------------------------

    def apply_move(self,action):

        self.state,won = game_quixo.move(self.state,action,self.player)

        if won:
            self.message = "Player wins!"
            return

        self.player *= -1
        self.selected = None

        pygame.time.wait(300)

        self.agent_move()

    # --------------------------
    # AGENT MOVE
    # --------------------------

    def agent_move(self):

        if self.net is None:
            return

        self.message = "Agent thinking..."

        pygame.display.flip()

        board = game_quixo.decode_board(self.state)

        batch = model_quixo.state_lists_to_batch([board],[self.player],"cpu")

        logits,_ = self.net(batch)

        probs = torch.softmax(logits,dim=1).detach().numpy()[0]

        legal = game_quixo.possible_moves(self.state,self.player)

        mask = np.zeros(game_quixo.N_ACTIONS)
        mask[legal] = 1

        probs *= mask
        probs /= probs.sum()

        action = np.argmax(probs)

        self.state,won = game_quixo.move(self.state,action,self.player)

        if won:
            self.message = "Agent wins!"
            return

        self.player *= -1
        self.message = "Your turn"

    # --------------------------
    # MOUSE CLICK
    # --------------------------

    def click(self,pos):

        x,y = pos

        c = (x-MARGIN)//CELL
        r = (y-MARGIN)//CELL

        if r<0 or r>=5 or c<0 or c>=5:
            return

        self.select_cube(r,c)


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
                app.message = "You start"

            if event.key == pygame.K_2:
                app.player = -1
                app.message = "Agent starts"
                app.agent_move()

            app.push_direction(event.key)

    app.draw()

    clock.tick(60)
