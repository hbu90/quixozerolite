import os
import time
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from lib import model_quixo


MAX_STEPS = 10_000
LEARNING_RATE = 0.2
GAMMA = 0.99
EPSILON = 0.1
PLAY_EPISODES = 25
SAVE_EVERY_N_STEPS = 500


def run_episode(net_weights):
    """
    Each worker:
    - reconstructs its own network
    - runs 1 full self-play episode
    - returns (result, move_count)
    """
    td_net = model_quixo.NTupleNetwork()
    for tup, w in zip(td_net.tuples, net_weights):
        tup.weights = w.copy()

    return model_quixo.play_game_train(
        td_net, alpha=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON
    )


def save_net(td_net, path):
    np.save(
        path,
        np.array([tup.weights for tup in td_net.tuples], dtype=object),
        allow_pickle=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)

    net = model_quixo.NTupleNetwork()
    step_idx = 0

    pool = Pool(processes=cpu_count() - 1)

    try:
        while step_idx < MAX_STEPS:
            print(f"\nSTEP {step_idx} — SELF PLAY")
            t0 = time.time()

            net_snapshot = np.array(
                [tup.weights.copy() for tup in net.tuples], dtype=object
            )

            results = list(
                tqdm(
                    pool.imap(run_episode, [net_snapshot] * PLAY_EPISODES),
                    total=PLAY_EPISODES,
                )
            )

            total_moves = sum(m for _, m in results)
            wins = sum(1 for r, _ in results if r == 1)

            dt = time.time() - t0

            print(f"Time: {dt:.2f}s")
            print(f"Moves/sec: {total_moves / dt:.2f}")
            print(f"Win rate (P1): {wins / PLAY_EPISODES:.2f}")

            if step_idx % SAVE_EVERY_N_STEPS == 0:
                save_path = os.path.join(saves_path, f"td_agent_step_{step_idx}.npy")
                save_net(net, save_path)
                print(f"Saved model → {save_path}")

            step_idx += 1

    finally:
        pool.close()
        pool.join()
