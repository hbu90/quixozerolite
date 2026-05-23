import os
import time
import random
import argparse
import collections
from tqdm import tqdm

from lib import game_quixo, model_quixo, mcts_quixo
from lib.agents_quixo import MCTSAgent
from evaluate import play_matches, win_ratio_from_results

import torch
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.tensorboard import SummaryWriter


class TargetNet:
    """
    Keeps a separate copy of the model which can be periodically synced from the online model.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.target_model = type(model)(
            input_shape=model.input_shape, actions_n=model.actions_n
        )
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())


PLAY_EPISODES = 25
MCTS_ITERATIONS = 12
MCTS_SIMULATION_SIZE = 5
REPLAY_BUFFER = 50000
LEARNING_RATE = 0.003
BATCH_SIZE = 128
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 5000
BEST_NET_WIN_RATIO = 0.55
EVALUATE_EVERY_STEP = 5
EVALUATION_ROUNDS = 40
MOVES_BEFORE_TAU_0 = 15
MAX_STEPS = 200


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

    net = model_quixo.Net(
        input_shape=model_quixo.OBS_SHAPE, actions_n=game_quixo.N_ACTIONS
    ).to(train_device)
    best_net = TargetNet(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    step_idx = 0
    best_idx = 0
    mcts_1 = mcts_quixo.MCTS()
    mcts_2 = mcts_quixo.MCTS()

    agent1 = MCTSAgent(net, mcts_1, MCTS_ITERATIONS, MCTS_SIMULATION_SIZE, train_device)
    agent2 = MCTSAgent(net, mcts_2, MCTS_ITERATIONS, MCTS_SIMULATION_SIZE, train_device)

    while step_idx < MAX_STEPS:
        t = time.time()
        prev_mcts_leaves = len(mcts_1) + len(mcts_2)
        total_game_moves = 0
        winner_count = 0

        print("SELF-PLAY")
        t0 = time.time()

        for episode in tqdm(range(PLAY_EPISODES)):
            print(f"EPISODE: {episode}")
            mcts_1.clear()
            mcts_2.clear()
            net1_result, moves = model_quixo.play_game(
                replay_buffer,
                agent1,
                agent2,
                moves_before_tau_0=MOVES_BEFORE_TAU_0,
            )
            total_game_moves += moves
            winner_count += net1_result

        print("SELF PLAY:", time.time() - t0)
        writer.add_scalar("replay_buffer_size", len(replay_buffer), step_idx)
        writer.add_scalar("mcts_tree_size_agent1", len(mcts_1), step_idx)
        writer.add_scalar("mcts_tree_size_agent2", len(mcts_2), step_idx)
        t1 = time.time()

        mcts_leaves = (len(mcts_1) + len(mcts_2)) - prev_mcts_leaves
        dt = time.time() - t
        moves_per_second = total_game_moves / dt
        mcts_leaves_per_second = mcts_leaves / dt
        avg_game_moves = total_game_moves / PLAY_EPISODES
        avg_winner_count = winner_count / PLAY_EPISODES
        writer.add_scalar("moves_per_second", moves_per_second, step_idx)
        writer.add_scalar("mcts_leaves_per_second", mcts_leaves_per_second, step_idx)
        writer.add_scalar("avg_game_moves", avg_game_moves, step_idx)
        writer.add_scalar("avg_winner_count", avg_winner_count, step_idx)
        print(
            "Step %d, avg_game_moves %3d, mcts_leaves %4d, moves_per_second %5.2f,"
            "mcts_leaves_per_second %6.2f, avg_winner_count %1d, best_idx %d, replay %d"
            % (
                step_idx,
                avg_game_moves,
                mcts_leaves,
                moves_per_second,
                mcts_leaves_per_second,
                avg_winner_count,
                best_idx,
                len(replay_buffer),
            )
        )
        step_idx += 1

        if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
            continue

        sum_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0

        for _ in tqdm(range(TRAIN_ROUNDS)):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
            batch_states_lists = [
                game_quixo.decode_board(state) for state in batch_states
            ]
            states_v = model_quixo.states_to_tensor_batch(
                batch_states_lists, batch_who_moves, train_device
            )

            optimizer.zero_grad()
            probs_v = torch.FloatTensor(batch_probs).to(train_device)
            values_v = torch.FloatTensor(batch_values).to(train_device)
            out_logits_v, out_values_v = net(states_v)

            loss_value_v = functional.mse_loss(out_values_v.squeeze(-1), values_v)
            loss_policy_v = -functional.log_softmax(out_logits_v, dim=1) * probs_v
            loss_policy_v = loss_policy_v.sum(dim=1).mean()

            loss_v = loss_policy_v + loss_value_v
            loss_v.backward()
            optimizer.step()
            sum_loss += loss_v.item()
            sum_value_loss += loss_value_v.item()
            sum_policy_loss += loss_policy_v.item()

        mcts_1.clear()
        mcts_2.clear()

        print("TRAINING:", time.time() - t1)
        t2 = time.time()

        writer.add_scalar("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
        writer.add_scalar("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
        writer.add_scalar("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)

        if step_idx % EVALUATE_EVERY_STEP == 0:
            eval_agent = MCTSAgent(
                net,
                mcts_quixo.MCTS(),
                MCTS_ITERATIONS,
                MCTS_SIMULATION_SIZE,
                train_device,
            )
            best_agent = MCTSAgent(
                best_net.target_model,
                mcts_quixo.MCTS(),
                MCTS_ITERATIONS,
                MCTS_SIMULATION_SIZE,
                train_device,
            )
            results = play_matches(
                eval_agent,
                best_agent,
                "eval_agent",
                "best_agent",
                games=EVALUATION_ROUNDS,
            )
            win_ratio = win_ratio_from_results(results)
            print("Net evaluated, win ratio = %.2f" % win_ratio)
            writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
            if win_ratio > BEST_NET_WIN_RATIO:
                print("Net is better than cur best, sync")
                best_net.sync()
                best_idx += 1
                file_name = os.path.join(
                    saves_path, "best_%03d_%05d.dat" % (best_idx, step_idx)
                )
                torch.save(net.state_dict(), file_name)
                mcts_1.clear()
                mcts_2.clear()
