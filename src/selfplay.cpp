#include "mz/selfplay.hpp"
#include "mz/board.hpp"
#include "mz/mcts.hpp"

namespace mz {

GameHistory playSelfPlayGame(const MuZeroNetwork& network, const SelfPlayConfig& config,
                             std::mt19937& rng) {
    MCTSConfig searchConfig;
    searchConfig.numSimulations = config.numSimulations;
    searchConfig.addRootNoise = true;
    searchConfig.dirichletAlpha = config.dirichletAlpha;
    searchConfig.dirichletEpsilon = config.dirichletEpsilon;

    Board board;
    GameHistory game;
    int ply = 0;

    // One search object for the whole game. Its node arena and inference
    // buffers are reused across moves; a fresh MCTS per ply would throw
    // them away and reallocate.
    MCTS mcts(network, searchConfig, rng);

    while (!board.isTerminal()) {
        float temperature = (ply < config.temperatureMoves) ? 1.0f : 0.0f;
        MCTSResult result = mcts.run(board, temperature);

        game.observations.push_back(board.encode());
        game.actions.push_back(result.selectedMove);
        game.searchPolicies.push_back(result.visitDistribution);
        // AlphaZero has no use for this and does not store it. Here it is
        // the source the n-step value target bootstraps from.
        game.searchValues.push_back(result.rootValue);
        game.rewards.push_back(0.0f);

        Cell mover = board.playerToMove();
        board = board.applyMove(result.selectedMove);
        ++ply;

        if (board.isTerminal()) {
            // Reward is in the perspective of the player who just moved.
            Outcome outcome = board.outcome();
            float reward = 0.0f;
            if (outcome != Outcome::Draw) {
                bool moverWon = (outcome == Outcome::XWins && mover == Cell::X) ||
                                (outcome == Outcome::OWins && mover == Cell::O);
                reward = moverWon ? 1.0f : -1.0f;
            }
            game.rewards.back() = reward;
        }
    }

    return game;
}

} // namespace mz
