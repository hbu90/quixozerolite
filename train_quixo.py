import os
import time
import numpy as np
import argparse
from tqdm import tqdm

from lib import model_quixo


MAX_STEPS = 10_000
LEARNING_RATE = 0.20
GAMMA = 0.99
EPSILON_DECAY_FACTOR = 0.95
PLAY_EPISODES = 25
SAVE_EVERY_N_STEPS = 1_000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)

    net = model_quixo.NTupleNetwork()
    step_idx = 1

    while step_idx <= MAX_STEPS:
        print(f"\nSTEP {step_idx} — SELF PLAY")
        t0 = time.time()
        total_game_moves = 0
        winner_count = 0

        EPSILON_DECAY_FACTOR *= 0.999

        for _ in tqdm(range(PLAY_EPISODES)):
            result, moves = model_quixo.play_game_train(
                net,
                alpha=LEARNING_RATE,
                gamma=GAMMA,
                epsilon_decay=EPSILON_DECAY_FACTOR,
            )

            total_game_moves += moves

            if result == 1:
                winner_count += 1

        dt = time.time() - t0

        print(f"Time: {dt:.2f}s")

        if step_idx % SAVE_EVERY_N_STEPS == 0:
            save_path = os.path.join(saves_path, f"td_agent_step_{step_idx}.npz")
            save_data = {}

            for i, tup in enumerate(net.tuples):
                save_data[f"tuple_{i}_weights"] = tup.weights
                save_data[f"tuple_{i}_positions"] = np.array(tup.positions)

            np.savez(save_path, **save_data)
            print(f"Saved model → {save_path}")

        step_idx += 1
