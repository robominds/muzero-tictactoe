#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/minimax.hpp"
#include "mz/network.hpp"

namespace {

void printGrid(const mz::Board& board) {
    const char* symbols[3] = {".", "X", "O"};
    for (int row = 0; row < 3; ++row) {
        std::printf("    ");
        for (int col = 0; col < 3; ++col) {
            std::printf("%s ", symbols[static_cast<int>(board.cellAt(row * 3 + col))]);
        }
        std::printf("\n");
    }
}

void playTranscript(const mz::MuZeroNetwork& network, bool networkPlaysX, int numSimulations) {
    std::printf("\n=== network plays %s ===\n", networkPlaysX ? "X" : "O");

    mz::MCTSConfig config;
    config.numSimulations = numSimulations;
    config.addRootNoise = false;
    std::mt19937 rng(20260910);

    mz::Board board;
    int ply = 0;
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == mz::Cell::X) == networkPlaysX;
        printGrid(board);
        if (networkTurn) {
            mz::MCTS mcts(network, config, rng);
            mz::MCTSResult result = mcts.run(board, 0.0f);
            std::printf("  ply %d  network -> %d   rootValue=%+.3f  depth=%d\n", ply,
                        result.selectedMove, result.rootValue, result.maxDepth);
            std::printf("  visits:");
            for (int a = 0; a < 9; ++a) {
                if (board.isLegalMove(a)) std::printf("  %d:%.2f", a, result.visitDistribution[a]);
            }
            std::printf("\n");
            board = board.applyMove(result.selectedMove);
        } else {
            int move = mz::minimaxBestMove(board);
            std::printf("  ply %d  minimax -> %d\n", ply, move);
            board = board.applyMove(move);
        }
        ++ply;
    }

    printGrid(board);
    mz::Outcome outcome = board.outcome();
    const char* verdict = "draw";
    if (outcome != mz::Outcome::Draw) {
        bool networkWon = (outcome == mz::Outcome::XWins && networkPlaysX) ||
                          (outcome == mz::Outcome::OWins && !networkPlaysX);
        verdict = networkWon ? "network WON (impossible vs perfect play -- check minimax)" : "network LOST";
    }
    std::printf("  result: %s\n", verdict);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: diag_eval <checkpoint-path> [simulations]\n");
        return 1;
    }
    int numSimulations = argc >= 3 ? std::atoi(argv[2]) : 150;

    mz::MuZeroNetwork network(1);
    try {
        network.load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    playTranscript(network, true, numSimulations);
    playTranscript(network, false, numSimulations);
    return 0;
}
