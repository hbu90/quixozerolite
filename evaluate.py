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

TOURNAMENT_MCTS_SIMULATION_SIZE = 200
MODEL_FOLDER = "20260902"
MODEL_PATH = f"saves/{MODEL_FOLDER}"


@dataclass
class TournamentPlayer:
    name: str
    agent: (
        agents_quixo.MCTSAgent
        | agents_quixo.TDAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    )
    elo: float = 0.0


def load_model(path: str) -> model_quixo.NTupleNetwork:
    """
    Load an N-tuple network from a saved checkpoint.

    Args:
        path (str): Path to the saved model checkpoint.
    Returns:
        model_quixo.NTupleNetwork: The loaded N-tuple network.
    Raises:
        KeyError: If a required weights or positions array is missing.
        ValueError: If tuple positions in the checkpoint do not match the network's tuple positions.
    """
    net = model_quixo.NTupleNetwork()

    with np.load(path) as data:
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
        | agents_quixo.TDAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    ),
    player_b_agent: (
        agents_quixo.MCTSAgent
        | agents_quixo.TDAgent
        | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent
        | agents_quixo.GreedyWinAgent
    ),
    player_a_name: str,
    player_b_name: str,
    games: int,
) ->tuple[list[tuple[str, str, float]], float]:
    """
    Play a series of games between two agents. Agents alternate which player moves first between games. MCTS trees are
    cleared before each game to prevent information carrying between games.

    Args:
        player_a_agent (agents_quixo.MCTSAgent | agents_quixo.TDAgent | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent | agents_quixo.GreedyWinAgent): Agent controlling player A.
        player_b_agent (agents_quixo.MCTSAgent | agents_quixo.TDAgent | agents_quixo.RandomAgent
        | agents_quixo.WinBlockAgent | agents_quixo.GreedyWinAgent): Agent controlling player B.
        player_a_name (str): Name used to identify player A in the results.
        player_b_name (str): Name used to identify player B in the results.
        games (int): Number of games to play.
    Returns:
        tuple[list[tuple[str, str, float]], float]: A tuple containing the individual game results and the percentage
        win rate for player A. Each game result is represented as (player_a_name, player_b_name, result).
    """
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

    player_a_wins = sum(result == 1 for _, _, result in results)
    win_rate = player_a_wins / games * 100

    return results, win_rate


def tournament(
    player_model_paths: list[str],
    games_per_pair: int = 100,
) -> tuple[list[tuple[str, str, float]], dict[str, float]]:
    """
    Run a round-robin tournament between all configured players. Each supplied model is evaluated using both its
    TD agent and MCTS agent. Random, GreedyWin, and WinBlock baseline agents are also included.

    Args:
        player_model_paths (list[str]): Paths to the saved model checkpoints to evaluate.
        games_per_pair (int): Number of games played for each pair of players.
    Returns:
        tuple[list[tuple[str, str, float]], dict[str, float]]: A tuple containing all individual game results and a
        dictionary of win rates for each player pairing.
    """
    players: list[TournamentPlayer] = []
    win_rates = {}

    for path in player_model_paths:
        basename = os.path.basename(path)
        net = load_model(path)
        td_agent = agents_quixo.TDAgent(
            net
        )
        td_name = f"{os.path.splitext(basename)[0]}"
        players.append(
            TournamentPlayer(name=td_name, agent=td_agent)
        )
        mcts_agent = agents_quixo.MCTSAgent(
            net,
            mcts_quixo.MCTS(),
            TOURNAMENT_MCTS_SIMULATION_SIZE,
        )
        mcts_name = f"mcts_wrapper_{os.path.splitext(basename)[0]}"
        players.append(
            TournamentPlayer(name=mcts_name, agent=mcts_agent)
        )

    random_player = agents_quixo.RandomAgent()
    greedy_win_player = agents_quixo.GreedyWinAgent()
    win_block_player = agents_quixo.WinBlockAgent()

    players.append(TournamentPlayer(name="RandomPlayer", agent=random_player))
    players.append(TournamentPlayer(name="GreedyWinPlayer", agent=greedy_win_player))
    players.append(TournamentPlayer(name="WinBlockPlayer", agent=win_block_player))

    pairs: list[tuple[int, int]] = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pairs.append((i, j))

    game_results = []
    for i, j in pairs:
        m1 = players[i]
        m2 = players[j]

        print(f"\nStarting {m1.name} vs {m2.name}")

        m1_v_m2_results, win_rate = play_matches(
            m1.agent, m2.agent, m1.name, m2.name, games=games_per_pair
        )

        print(f"Finished {m1.name} vs {m2.name}. Win rate was {win_rate}")
        win_rates[f"{m1.name}_vs_{m2.name}"] = win_rate
        game_results.extend(m1_v_m2_results)

    return game_results, win_rates


def expected_score(r_a: float, r_b: float) -> float:
    """
    Calculate the expected score of one player against another.

    Args:
        r_a (float): Elo rating of player A.
        r_b (float): Elo rating of player B.
    Returns:
        float: The expected score for player A, between 0 and 1.
    """
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


def update_elo(
    r_a: float, r_b: float, score_a: float, k: float = 32.0
) -> tuple[float, float]:
    """
    Update the Elo ratings of two players after a game.

    Args:
        r_a (float): Current Elo rating of player A.
        r_b (float): Current Elo rating of player B.
        score_a (float): Actual score achieved by player A, where 1 is a win,
            0.5 is a draw, and 0 is a loss.
        k (float): Elo update factor controlling the size of rating changes.
    Returns:
        tuple[float, float]: A tuple containing the updated Elo ratings for players A and B.
    """
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
    """
    Calculate Elo ratings from a sequence of game results. Games are processed sequentially, with each game's result
    updating the ratings before the next game is processed.

    Args:
        games (list[tuple[str, str, float]]): Game results as tuples of (player_1, player_2, result), where result is
        positive for a player 1 win, negative for a player 1 loss, and zero for a draw.
        k (float): Elo update factor controlling the size of rating changes.
        start_rating (float): Initial Elo rating assigned to each player.
    Returns:
        dict[str, float]: A dictionary mapping each player name to their final Elo rating.
    """
    ratings = {}

    for p1, p2, result in games:
        if p1 not in ratings:
            ratings[p1] = start_rating
        if p2 not in ratings:
            ratings[p2] = start_rating

        if result > 0.5:
            score_p1 = 1.0
        elif result < -0.5:
            score_p1 = 0.0
        else:
            score_p1 = 0.5

        ratings[p1], ratings[p2] = update_elo(ratings[p1], ratings[p2], score_p1, k=k)

    return ratings


def offline_elo(
    games: list[tuple[str, str, float]],
    n_shuffles: int = 200,
    k: float = 32.0,
    start_rating: float = 1000.0,
    seed: int = 0,
) -> dict[str, float]:
    """
    Estimate Elo ratings by averaging ratings across shuffled game orders. The games are repeatedly shuffled and Elo
    ratings are calculated for each ordering. The returned rating for each player is the mean across all shuffled runs,
    reducing the influence of the original game ordering.

    Args:
        games (list[tuple[str, str, float]]): Game results as tuples of (player_1, player_2, result).
        n_shuffles (int): Number of random game orderings used to calculate Elo.
        k (float): Elo update factor controlling the size of rating changes.
        start_rating (float): Initial Elo rating assigned to each player.
        seed (int): Random seed used when shuffling the games.
    Returns:
        dict[str, float]: A dictionary mapping each player name to their mean Elo rating across all shuffled game
        orders.
    """
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
    tournament_results, win_rate_results = tournament(
        player_model_paths=model_paths,
        games_per_pair=200,
    )

    with open(f"win_rates_{MODEL_FOLDER}.json", "w") as f:
        json.dump(win_rate_results, f, indent=4)

    with open(f"tournament_results_{MODEL_FOLDER}.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerows(tournament_results)

    offline_elo_ratings = offline_elo(tournament_results, n_shuffles=20, k=32)

    with open(f"elo_ratings_{MODEL_FOLDER}.json", "w") as f:
        json.dump(offline_elo_ratings, f, indent=4)

    for n, r in sorted(offline_elo_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{n:20s} {r:.1f}")
