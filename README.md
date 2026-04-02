# Quixo Reinforcement Learning Agent

A Python project implementing a Reinforcement Learning agent to play the board game **Quixo**, following the approach of AlphaZero, using Monte-Carlo Tree Search (MCTS) and neural networks.

## Features

- Self-play training using MCTS and a PyTorch neural network.
- Encode/decode Quixo boards for efficient storage and computation.
- TensorBoard logging for tracking training progress.
- Save and load trained models (`.dat` files).
- Optional GUI interface to play against trained agent.

## Installation

1. Clone the repository:

```bash
git clone ...
cd quixo-rl
```
2. Install dependencies:

```bash
poetry install
```

## Quixo

Quixo (KWIK-SOH) is a board game played on a 5x5 grid, which contains 25 pieces (cubes). 
The cubes have four blank faces, and one face with an X and one face with an O. 
The game has some resemblance to Tic-Tac-Toe, but with more complexity.

Each player take it in turns to select a cube that is either blank or bearing their symbol (X or O) from
the border pieces of the 5x5 board and then returns it to the board bearing the player's symbol, 
**by pushing it into one of the rows it was taken from**. The game is won when either player has five of their 
pieces either horizontally, vertically, or diagonally.

The game of Quixo lends itself to the AlphaZero methodology, as it is a zero-sum game of perfect 
information (each player can see the entire board and the moves each player makes). The state space is large, with
2 * 3^25..

Research into the game of Quixo has stated that with perfect play from both players, Quixo is a draw game ("Quixo Is Solved")
and that neither player can force a win under perfect play. Another paper ("QUIXO is EXPTIME-complete") 


## AlphaZero

AlphaGo Zero ("Mastering the game of Go without human knowledge" (Silver et al., 2017)) was trained using self-play reinforcement learning, without any supervision. It used a single neural network (rather than 
previous iterations, AlphaGo Fan and AlphaGo Lee, which used distinct policy and value networks). The input to the neural network is the state of the board, s, and the outputs are the probabilities of each action 
and a scalar value (which estimates the probability of the current player winning from the position s). 

AlphaZero ("Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", (Silver et al., 2017))
was the generalisation of the AlphaGo Zero approach. 

### MCTS and Self-Play

The MCTS uses the neural network to guide the simulations.  Within the MCTS, visit counts, values, average values, and probabilities are 
stored for each state. The MCTS traverses the tree from a given root node, using the UCB score to select the action, 
until a leaf node is found. A leaf node is a node that has not been explored by the MCTS previously. 

When a leaf node is found, it is added to a queue for eventual expansion. This is done in batches for efficiency.
The neural network is used to evaluate and returns values and probabilities for each of the leaf nodes, which are then passed 
into the MCTS statistics. 

Backup is the phase where the MCTS statistics are updated i.e., the visit counts for each node are updated and the values/average values are 
also updated. 

An action is selected after all of this by querying the visit counts stored in the MCTS. This is dependent on τ, a temperature parameter
that can control the degree of exploration within the MCTS policy distribution. 

**SELECTION - EXPANSION - BACKUP - PLAY**

UCB score - 

In AlphaGo Zero and AlphaZero, no rollouts are used (which was not the case for AlphaGo Fan and AlphaGo Lee).
A rollout is where the game is played onwards from the leaf node to estimate the strength of the position.

![Diagram](images/mcts.png)

Self-play is where the program plays against itself, instead of against a human or another program. The program is provided with complete information of the game structure and rules, but is not given 
any heuristics or rules on what policy to use. 

MCTS is described as a tool for both policy improvement and policy evaluation ("Mastering the game of Go without human knowledge" (Silver et al., 2017)), 
as these searches are iterated within self-play and used to select each move. Subsequently, the neural network is then trained with the results of the more recent 
self-play data which includes moves and the winner (z). 

![Diagram](images/nn_training.png)

