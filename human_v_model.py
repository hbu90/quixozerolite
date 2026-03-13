#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from lib import game, model, mcts

MCTS_SEARCHES = 50
MCTS_BATCH_SIZE = 8


def print_board(state_int):
    board = game.render(state_int)

    # Adjust symbols to match what render() actually returns
    SYMBOLS = {
        ' ': "·",   # empty
        0: "🔴",    # one player
        1: "🟡",    # other player
    }

    print()
    print("  0   1   2   3   4   5   6")
    print("┌───┬───┬───┬───┬───┬───┬───┐")

    for row in board:
        print("│", end="")
        for cell in row:
            # Convert string digits to int if needed
            if isinstance(cell, str) and cell.isdigit():
                cell = int(cell)
            print(f" {SYMBOLS[cell]} │", end="")
        print()
        print("├───┼───┼───┼───┼───┼───┼───┤")

    print("└───┴───┴───┴───┴───┴───┴───┘")
    print()


def human_move(state_int):
    legal = game.possible_moves(state_int)
    print("Legal moves:", legal)

    while True:
        try:
            move = int(input("Your move: "))
            if move in legal:
                return move
            print("Invalid move.")
        except ValueError:
            print("Enter a column number.")


def model_move(state_int, player, net, device):
    mcts_store = mcts.MCTS()

    # run MCTS simulations
    mcts_store.search_batch(
        count=MCTS_SEARCHES,
        batch_size=MCTS_BATCH_SIZE,
        state_int=state_int,
        player=player,
        net=net,
        device=device
    )

    probs, _ = mcts_store.get_policy_value(state_int, tau=0)
    action = int(np.argmax(probs))
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained model")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--human-first", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    # load trained network
    net = model.Net(model.OBS_SHAPE, game.GAME_COLS)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device)
    net.eval()

    state = game.INITIAL_STATE
    current_player = game.PLAYER_BLACK if args.human_first else game.PLAYER_WHITE
    human_player = current_player

    print("Game start!")
    print("You are player", human_player)
    print_board(state)

    while True:
        if current_player == human_player:
            action = human_move(state)
        else:
            print("Model is thinking...")
            action = model_move(state, current_player, net, device)

        state, won = game.move(state, action, current_player)
        print_board(state)

        if won:
            if current_player == human_player:
                print("You win 🎉")
            else:
                print("Model wins 🤖")
            break

        if len(game.possible_moves(state)) == 0:
            print("Draw!")
            break

        current_player = 1 - current_player


if __name__ == "__main__":
    main()