import os
import time
import ptan
import random
import argparse
import collections
from tqdm import tqdm

from lib import game_quixo, model_quixo, mcts_quixo
from lib.agents_quixo import MCTSAgent
from evaluate import play_matches, win_ratio_from_results

from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as functional

PLAY_EPISODES = 40
MCTS_ITERATIONS = 60
MCTS_SIMULATION_SIZE = 24
REPLAY_BUFFER = 50000
LEARNING_RATE = 0.003
BATCH_SIZE = 128
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 5000
BEST_NET_WIN_RATIO = 0.55
EVALUATE_EVERY_STEP = 5
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 15
MAX_STEPS = 500

# PLAY_EPISODES = 2
# MCTS_ITERATIONS = 5
# MCTS_SIMULATION_SIZE = 2
# REPLAY_BUFFER = 1000
# LEARNING_RATE = 0.01
# BATCH_SIZE = 32
# TRAIN_ROUNDS = 1
# MIN_REPLAY_TO_TRAIN = 200
# BEST_NET_WIN_RATIO = 0.55
# EVALUATE_EVERY_STEP = 10
# EVALUATION_ROUNDS = 3
# STEPS_BEFORE_TAU_0 = 8
# MAX_STEPS = 200


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
    best_net = ptan.agent.TargetNet(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    step_idx = 0
    best_idx = 0
    mcts_1 = mcts_quixo.MCTS()
    mcts_2 = mcts_quixo.MCTS()

    agent1 = MCTSAgent(net, mcts_1, MCTS_ITERATIONS, MCTS_SIMULATION_SIZE, train_device)
    agent2 = MCTSAgent(net, mcts_2, MCTS_ITERATIONS, MCTS_SIMULATION_SIZE, train_device)

    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while step_idx < MAX_STEPS:
            t = time.time()
            prev_nodes = len(mcts_1) + len(mcts_2)
            game_steps = 0

            print("SELF-PLAY")
            t0 = time.time()

            for _ in tqdm(range(PLAY_EPISODES)):
                _, steps = model_quixo.play_game(
                    replay_buffer,
                    agent1,
                    agent2,
                    steps_before_tau_0=STEPS_BEFORE_TAU_0,
                )
                game_steps += steps

            print("SELF PLAY:", time.time() - t0)
            tb_tracker.track("replay_size", len(replay_buffer), step_idx)
            tb_tracker.track("mcts_tree_size_agent1", len(mcts_1), step_idx)
            tb_tracker.track("mcts_tree_size_agent2", len(mcts_2), step_idx)
            t1 = time.time()

            game_nodes = (len(mcts_1) + len(mcts_2)) - prev_nodes
            dt = time.time() - t
            speed_steps = game_steps / dt
            speed_nodes = game_nodes / dt
            tb_tracker.track("speed_steps", speed_steps, step_idx)
            tb_tracker.track("speed_nodes", speed_nodes, step_idx)
            print(
                "Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d, replay %d"
                % (
                    step_idx,
                    game_steps,
                    game_nodes,
                    speed_steps,
                    speed_nodes,
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

            print("TRAINING:", time.time() - t1)
            t2 = time.time()

            tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)

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
