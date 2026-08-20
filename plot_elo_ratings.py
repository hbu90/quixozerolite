import json
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("elo_ratings.json", "r") as f:
    elo_ratings = json.load(f)

rows = []

for key, elo in elo_ratings.items():
    match = re.search(r"_(\d+)$", key)
    if match:
        rows.append(
            {"step": int(match.group(1)), "elo": elo, "player": key[: match.start()]}
        )

df = pd.DataFrame(rows).sort_values("step")

reference_players = [
    ("RandomPlayer", "grey"),
    ("WinBlockPlayer", "orange"),
    ("GreedyWinPlayer", "red"),
]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

sns.lineplot(data=df, x="step", y="elo", marker="o", linewidth=2, label="Training Elo")

for player, colour in reference_players:
    if player in elo_ratings:
        plt.axhline(
            elo_ratings[player],
            color=colour,
            linestyle="--",
            linewidth=2,
            label=f"{player} ({elo_ratings[player]:.0f})",
        )

plt.xlabel("Training Step")
plt.ylabel("Elo")
plt.title("Training Elo Progression")
plt.legend()
plt.tight_layout()
plt.show()
