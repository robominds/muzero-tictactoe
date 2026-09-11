#include <cstdio>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/network.hpp"

namespace {

void printBoard(const mz::Board& board) {
    const char* symbols[3] = {".", "X", "O"};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            std::printf("%s ", symbols[static_cast<int>(board.cellAt(row * 3 + col))]);
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: play_cli <checkpoint-path>\n");
        return 1;
    }

    mz::MuZeroNetwork network(1);
    try {
        network.load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::printf("You are X. Enter a move as a number 0-8 (see grid below).\n");
    std::printf("0 1 2\n3 4 5\n6 7 8\n\n");

    mz::MCTSConfig config;
    config.numSimulations = 300;
    config.addRootNoise = false;
    std::mt19937 rng(std::random_device{}());

    mz::Board board;
    while (!board.isTerminal()) {
        printBoard(board);
        if (board.playerToMove() == mz::Cell::X) {
            int move = -1;
            while (true) {
                std::printf("Your move: ");
                if (!(std::cin >> move) || !board.isLegalMove(move)) {
                    std::printf("Invalid move, try again.\n");
                    std::cin.clear();
                    std::cin.ignore(10000, '\n');
                    if (std::cin.eof()) {
                        std::printf("\nInput ended, exiting.\n");
                        return 0;
                    }
                    continue;
                }
                break;
            }
            board = board.applyMove(move);
        } else {
            mz::MCTS mcts(network, config, rng);
            int move = mcts.run(board, 0.0f).selectedMove;
            std::printf("Agent plays %d\n", move);
            board = board.applyMove(move);
        }
    }

    printBoard(board);
    mz::Outcome outcome = board.outcome();
    if (outcome == mz::Outcome::Draw) std::printf("Draw.\n");
    else if (outcome == mz::Outcome::XWins) std::printf("You win!\n");
    else std::printf("Agent wins.\n");

    return 0;
}
