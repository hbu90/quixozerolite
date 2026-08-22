from dataclasses import dataclass
import lib.model_quixo as model_quixo
import lib.agents_quixo as agents_quixo
import lib.mcts_quixo as mcts_quixo
import os
import json
import random
import numpy as np
from pathlib import Path
import csv

TOURNAMENT_MCTS_ITERATIONS = 10
TOURNAMENT_MCTS_SIMULATION_SIZE = 4
MODEL_PATH = "saves/20260815"

@dataclass
class TournamentPlayer:
    name: str
    agent: (
        agents_quixo.MCTSAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    )
    elo: float = 0.0


def load_model(path: str) -> model_quixo.NTupleNetwork:
    net = model_quixo.NTupleNetwork()

    data = np.load(path)

    for i, tup in enumerate(net.tuples):
        weight_key = f"tuple_{i}_weights"
        pos_key = f"tuple_{i}_positions"

        if weight_key not in data or pos_key not in data:
            raise KeyError(f"Missing '{weight_key}' or '{pos_key}'.")

        if not np.array_equal(data[pos_key], tup.positions):
            raise ValueError(f"Tuple {i} positions do not match checkpoint.")

        tup.weights[:] = data[weight_key]

    return net


def play_matches(
    player_a_agent: (
        agents_quixo.MCTSAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    ),
    player_b_agent: (
        agents_quixo.MCTSAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    ),
    player_a_name: str,
    player_b_name: str,
    games: int,
) -> list[tuple[str, str, float]]:
    """ """
    results = []

    for game_idx in range(games):
        print(f"  Game {game_idx + 1}/{games}")
        player_a_first = game_idx % 2 == 0

        if isinstance(player_a_agent, agents_quixo.MCTSAgent):
            player_a_agent.mcts.clear()

        if isinstance(player_b_agent, agents_quixo.MCTSAgent):
            player_b_agent.mcts.clear()

        result, _ = model_quixo.play_game_full(
            agent1=player_a_agent,
            agent2=player_b_agent,
            moves_before_tau_0=0,
            agent1_plays_first=player_a_first,
        )

        results.append((player_a_name, player_b_name, result))

    return results


def win_ratio_from_results(results: list[tuple[str, str, float]]) -> float:
    n_a_wins = 0
    n_b_wins = 0

    for _, _, result in results:
        if result > 0.5:
            n_a_wins += 1
        elif result < -0.5:
            n_b_wins += 1

    denominator = n_a_wins + n_b_wins
    return 0.0 if denominator == 0 else n_a_wins / denominator


def tournament(
    player_model_paths: list[str],
    games_per_pair: int = 100,
) -> list[tuple[str, str, float]]:
    players = []

    for path in player_model_paths:
        basename = os.path.basename(path)
        net = load_model(path)
        agent = agents_quixo.MCTSAgent(
            net,
            mcts_quixo.MCTS(),
            TOURNAMENT_MCTS_ITERATIONS,
            TOURNAMENT_MCTS_SIMULATION_SIZE,
        )
        players.append(
            TournamentPlayer(name=os.path.splitext(basename)[0], agent=agent)
        )

    random_player = agents_quixo.RandomAgent()
    greedy_win_player = agents_quixo.GreedyWinAgent()
    win_block_player = agents_quixo.WinBlockAgent()

    players.append(TournamentPlayer(name="RandomPlayer", agent=random_player))
    players.append(TournamentPlayer(name="GreedyWinPlayer", agent=greedy_win_player))
    players.append(TournamentPlayer(name="WinBlockPlayer", agent=win_block_player))

    pairs = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pairs.append((i, j))

    game_results = []
    for i, j in pairs:
        m1 = players[i]
        m2 = players[j]

        print(f"\nStarting {m1.name} vs {m2.name}")

        m1_v_m2_results = play_matches(
            m1.agent, m2.agent, m1.name, m2.name, games=games_per_pair
        )

        print(f"Finished {m1.name} vs {m2.name}")

        game_results.extend(m1_v_m2_results)

    return game_results


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


def update_elo(
    r_a: float, r_b: float, score_a: float, k: float = 32.0
) -> tuple[float, float]:
    e_a = expected_score(r_a, r_b)
    e_b = expected_score(r_b, r_a)

    r_a_new = r_a + k * (score_a - e_a)
    r_b_new = r_b + k * ((1.0 - score_a) - e_b)

    return r_a_new, r_b_new


def compute_elo_from_games(
    games: list[tuple[str, str, float]],
    k: float = 32.0,
    start_rating: float = 1000.0,
) -> dict[str, float]:
    ratings = {}

    for p1, p2, score_p1 in games:
        if p1 not in ratings:
            ratings[p1] = start_rating
        if p2 not in ratings:
            ratings[p2] = start_rating

        ratings[p1], ratings[p2] = update_elo(ratings[p1], ratings[p2], score_p1, k=k)

    return ratings


def offline_elo(
    games: list[tuple[str, str, float]],
    n_shuffles: int = 200,
    k: float = 32.0,
    start_rating: float = 1000.0,
    seed: int = 0,
) -> dict[str, float]:
    rng = random.Random(seed)

    all_players = set()
    for p1, p2, _ in games:
        all_players.add(p1)
        all_players.add(p2)

    rating_samples = {p: [] for p in all_players}

    for _ in range(n_shuffles):
        shuffled = games[:]
        rng.shuffle(shuffled)

        ratings = compute_elo_from_games(shuffled, k=k, start_rating=start_rating)

        for p in all_players:
            rating_samples[p].append(ratings[p])

    avg_ratings = {p: float(np.mean(vals)) for p, vals in rating_samples.items()}
    return avg_ratings


if __name__ == "__main__":
    model_paths = [str(p) for p in Path(MODEL_PATH).glob("*.npz")]

    tournament_results = tournament(
        player_model_paths=model_paths,
        games_per_pair=100,
    )

    with open(f"tournament_results_{MODEL_PATH}.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerows(tournament_results)

    offline_elo_ratings = offline_elo(tournament_results, n_shuffles=20, k=32)

    with open(f"elo_ratings_{MODEL_PATH}.json", "w") as f:
        json.dump(offline_elo_ratings, f, indent=4)

    for n, r in sorted(offline_elo_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{n:20s} {r:.1f}")
