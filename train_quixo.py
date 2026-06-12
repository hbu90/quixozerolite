import os
import time
import numpy as np
import argparse
from tqdm import tqdm

from lib import model_quixo

import torch
from torch.utils.tensorboard import SummaryWriter


MAX_STEPS = 20
LEARNING_RATE = 0.01
GAMMA = 0.99
EPSILON = 0.1
PLAY_EPISODES = 25
EVALUATE_EVERY_STEP = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA"
    )
    args = parser.parse_args()
    if args.cuda and torch.cuda.is_available():
        train_device = torch.device("cuda")
    else:
        train_device = torch.device("cpu")

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment="-" + args.name)

    net = model_quixo.NTupleNetwork()
    step_idx = 0

    while step_idx < MAX_STEPS:
        t = time.time()
        total_game_moves = 0
        winner_count = 0

        print("SELF-PLAY")
        t0 = time.time()

        for _ in tqdm(range(PLAY_EPISODES)):
            result, moves = model_quixo.play_game_train(
                net, alpha=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON
            )

            total_game_moves += moves

            if result == 1:
                winner_count += 1

        print("SELF PLAY:", time.time() - t0)
        dt = time.time() - t
        moves_per_second = total_game_moves / dt
        avg_game_moves = total_game_moves / PLAY_EPISODES
        avg_winner_count = winner_count / PLAY_EPISODES
        writer.add_scalar("moves_per_second", moves_per_second, step_idx)
        writer.add_scalar("avg_game_moves", avg_game_moves, step_idx)
        writer.add_scalar("avg_winner_count", avg_winner_count, step_idx)

        if step_idx % 10 == 0:
            save_path = os.path.join(saves_path, f"td_agent_step_{step_idx}.npy")
            np.save(save_path, [tup.weights for tup in net.tuples])

            print(f"Saved to {save_path}")

        step_idx += 1
