// gpu_tictactoe.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BOARD_SIZE 9

// 0 = empty, 1 = X, -1 = O

// All 8 winning lines, in terms of indices into 3x3 board
__device__ __constant__ int WIN_LINES[8][3] = {
    {0,1,2}, {3,4,5}, {6,7,8}, // rows
    {0,3,6}, {1,4,7}, {2,5,8}, // cols
    {0,4,8}, {2,4,6}           // diagonals
};

__device__ int check_win(const int *board) {
    // returns 1 if X wins, -1 if O wins, 0 otherwise
    for (int l = 0; l < 8; ++l) {
        int a = WIN_LINES[l][0];
        int b = WIN_LINES[l][1];
        int c = WIN_LINES[l][2];
        int sum = board[a] + board[b] + board[c];
        if (sum == 3)  return 1;
        if (sum == -3) return -1;
    }
    return 0;
}

// Common device helper to score a move for a given player
__device__ int score_move(const int *board, int moveIdx, int player) {
    if (board[moveIdx] != 0) return -1000; // illegal move

    int tmp[BOARD_SIZE];
    #pragma unroll
    for (int i = 0; i < BOARD_SIZE; ++i) tmp[i] = board[i];

    tmp[moveIdx] = player;
    int winner = check_win(tmp);
    if (winner == player) return 100; // immediate win

    // check if this move blocks opponent's immediate win
    int opp = -player;
    int blocked = 0;
    for (int m = 0; m < BOARD_SIZE; ++m) {
        if (tmp[m] == 0) {
            tmp[m] = opp;
            if (check_win(tmp) == opp) blocked = 1;
            tmp[m] = 0;
        }
    }
    if (blocked) return 80; // good blocking move

    // weaker heuristic: prefer center, then corners, then edges
    int bonus = 0;
    if (moveIdx == 4) bonus = 10;                    // center
    else if (moveIdx == 0 || moveIdx == 2 ||
             moveIdx == 6 || moveIdx == 8) bonus = 5;// corners
    else bonus = 3;                                  // edges

    return bonus;
}

// Player A: greedy, uses the base scoring
__global__ void choose_move_playerA(const int *board, int player, int *scores) {
    int idx = threadIdx.x;
    if (idx >= BOARD_SIZE) return;
    scores[idx] = score_move(board, idx, player);
}

// Player B: similar but adds a simple bias to play differently
__global__ void choose_move_playerB(const int *board, int player, int *scores) {
    int idx = threadIdx.x;
    if (idx >= BOARD_SIZE) return;
    int base = score_move(board, idx, player);

    // simple “style” difference: prefer edges more aggressively,
    // slightly dislike corners; this changes tie-breaks often
    if (idx == 4) {
        base += 0; // neutral
    } else if (idx == 0 || idx == 2 || idx == 6 || idx == 8) {
        base -= 2; // corners less attractive
    } else {
        base += 4; // edges more attractive
    }

    scores[idx] = base;
}

void print_board(const int *board) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            int v = board[3*r + c];
            char ch = '.';
            if (v == 1)  ch = 'X';
            if (v == -1) ch = 'O';
            printf("%c ", ch);
        }
        printf("\n");
    }
    printf("\n");
}

int host_check_win(const int *board) {
    static const int lines[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };
    for (int l = 0; l < 8; ++l) {
        int a = lines[l][0], b = lines[l][1], c = lines[l][2];
        int sum = board[a] + board[b] + board[c];
        if (sum == 3)  return 1;
        if (sum == -3) return -1;
    }
    return 0;
}

bool board_full(const int *board) {
    for (int i = 0; i < BOARD_SIZE; ++i)
        if (board[i] == 0) return false;
    return true;
}

int main() {
    int board[BOARD_SIZE] = {0};

    int *d_board = nullptr;
    int *d_scores = nullptr;
    cudaMalloc(&d_board,  BOARD_SIZE * sizeof(int));
    cudaMalloc(&d_scores, BOARD_SIZE * sizeof(int));

    printf("GPU vs GPU Tic-Tac-Toe\n\n");

    int currentPlayer = 1; // 1 = X (Player A), -1 = O (Player B)
    int moveCount = 0;

    while (true) {
        printf("Current board:\n");
        print_board(board);

        // copy board to device
        cudaMemcpy(d_board, board, BOARD_SIZE * sizeof(int),
                   cudaMemcpyHostToDevice);

        // launch appropriate kernel
        if (currentPlayer == 1) {
            printf("GPU Player A (X) thinking...\n");
            choose_move_playerA<<<1, BOARD_SIZE>>>(d_board, currentPlayer, d_scores);
        } else {
            printf("GPU Player B (O) thinking...\n");
            choose_move_playerB<<<1, BOARD_SIZE>>>(d_board, currentPlayer, d_scores);
        }
        cudaDeviceSynchronize();

        int scores[BOARD_SIZE];
        cudaMemcpy(scores, d_scores, BOARD_SIZE * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // pick best move on host
        int bestIdx = -1;
        int bestScore = -100000;
        for (int i = 0; i < BOARD_SIZE; ++i) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        }

        if (bestIdx == -1 || board[bestIdx] != 0) {
            printf("No legal move found. Stopping.\n");
            break;
        }

        board[bestIdx] = currentPlayer;
        moveCount++;

        printf("Player %c played position %d (score %d)\n\n",
               (currentPlayer == 1 ? 'X' : 'O'), bestIdx, bestScore);

        int w = host_check_win(board);
        if (w == 1) {
            print_board(board);
            printf("GPU Player A (X) wins!\n");
            break;
        } else if (w == -1) {
            print_board(board);
            printf("GPU Player B (O) wins!\n");
            break;
        } else if (board_full(board)) {
            print_board(board);
            printf("Game is a draw.\n");
            break;
        }

        currentPlayer = -currentPlayer; // switch player
    }

    cudaFree(d_board);
    cudaFree(d_scores);
    cudaDeviceReset();
    return 0;
}
