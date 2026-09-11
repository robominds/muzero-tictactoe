#include "mz/eval.hpp"
#include <random>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/minimax.hpp"

namespace mz {

namespace {

int playOneGame(const MuZeroNetwork& network, bool networkPlaysX, int numSimulations,
                std::mt19937& rng) {
    MCTSConfig config;
    config.numSimulations = numSimulations;
    config.addRootNoise = false;   // the network's unperturbed best play

    Board board;
    // Reused across the whole game; see the note in selfplay.cpp.
    MCTS mcts(network, config, rng);
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == Cell::X) == networkPlaysX;
        int move;
        if (networkTurn) {
            move = mcts.run(board, 0.0f).selectedMove;
        } else {
            move = minimaxBestMove(board);
        }
        board = board.applyMove(move);
    }

    Outcome outcome = board.outcome();
    if (outcome == Outcome::Draw) return 0;
    bool networkWon = (outcome == Outcome::XWins && networkPlaysX) ||
                      (outcome == Outcome::OWins && !networkPlaysX);
    return networkWon ? 1 : -1;
}

} // namespace

EvalResult evaluateAgainstMinimax(const MuZeroNetwork& network, int gamesPerSide,
                                  int numSimulations) {
    EvalResult result;
    std::mt19937 rng(12345);   // fixed: evaluation should be reproducible
    auto record = [&](int outcome) {
        if (outcome == 1) result.wins++;
        else if (outcome == -1) result.losses++;
        else result.draws++;
    };
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, true, numSimulations, rng));
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, false, numSimulations, rng));
    return result;
}

} // namespace mz
