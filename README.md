GPU vs GPU Tic-Tac-Toe (CUDA)

This project implements a tic-tac-toe game where both players are CUDA kernels running on the GPU. The CPU acts as the referee: it maintains the board, alternates turns, launches the appropriate kernel for each GPU player, and checks for wins or draws.

Each GPU kernel evaluates all legal moves in parallel, one thread per square, and assigns a score to each possible move. The CPU reads back the scores and selects the best move for that player. The board is printed after every turn so you can watch the game unfold.

Game Logic
Board Representation

3×3 board stored as a 9-element integer array:

0 → empty

1 → Player A (X)

-1 → Player B (O)

Player Strategies

GPU Player A (X)
Greedy heuristic:

+100 for immediate win

+80 for blocking opponent win

Prefers center > corners > edges

GPU Player B (O)
Variant heuristic:

Same base scoring

Prefers edges more, penalizes corners

Produces different move choices from Player A

Kernel Workflow

CPU copies current board to GPU.

Kernel runs with 9 threads (one per board position).

Each thread:

Checks legality

Simulates the move

Scores it according to the player’s heuristic

CPU copies back scores and picks the highest.

CPU updates the board and prints it.

Game continues until a win or draw.

Build and Run
Compile
nvcc gpu_tictactoe.cu -o gpu_tictactoe

Run
./gpu_tictactoe


You will see each board state and which GPU selected which move.
