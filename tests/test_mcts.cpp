#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/network.hpp"

using namespace mz;

namespace {
bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }
} // namespace

void test_minmax_stats_passes_through_before_any_update() {
    MinMaxStats stats;
    assert(near(stats.normalize(0.7f), 0.7f));
    assert(near(stats.normalize(-2.0f), -2.0f));
}

void test_minmax_stats_normalizes_observed_range_to_unit_interval() {
    MinMaxStats stats;
    stats.update(-0.3f);
    stats.update(0.3f);
    assert(near(stats.normalize(-0.3f), 0.0f));
    assert(near(stats.normalize(0.3f), 1.0f));
    assert(near(stats.normalize(0.0f), 0.5f));
}

void test_minmax_stats_handles_a_single_observation() {
    MinMaxStats stats;
    stats.update(0.42f);
    assert(std::isfinite(stats.normalize(0.42f)));
}

void test_edge_q_negates_the_child_value() {
    // Q of the edge into a child, in the PARENT's perspective: the reward
    // the parent's player collects, plus the (negated) value of the
    // position it hands the opponent.
    assert(near(edgeQ(0.5f, 0.8f, 1.0f), -0.3f));
    assert(near(edgeQ(0.0f, -0.6f, 1.0f), 0.6f));
}

void test_exploration_term_shrinks_as_a_child_is_visited() {
    float first = explorationTerm(0.4f, 100, 0);
    float later = explorationTerm(0.4f, 100, 10);
    assert(first > later);
}

void test_exploration_term_scales_with_prior() {
    assert(explorationTerm(0.8f, 50, 3) > explorationTerm(0.2f, 50, 3));
}

void test_exploration_term_grows_with_parent_visits() {
    assert(explorationTerm(0.3f, 400, 5) > explorationTerm(0.3f, 100, 5));
}

void test_backup_alternates_sign_and_carries_reward() {
    // Three nodes. B is reached by a move that paid its mover +0.5, and the
    // leaf evaluates to +0.8 for whoever moves at B.
    BackupNode root{0.0f, 0, 0.0f};
    BackupNode a{0.0f, 0, 0.0f};
    BackupNode b{0.5f, 0, 0.0f};
    std::vector<BackupNode*> path = {&root, &a, &b};

    MinMaxStats stats;
    backupPath(path, 0.8f, 1.0f, stats);

    assert(b.visitCount == 1 && near(b.valueSum, 0.8f));
    // From A's player's view: collect 0.5, hand over a 0.8 position.
    assert(a.visitCount == 1 && near(a.valueSum, -0.3f));
    // And back to the root player, negated once more.
    assert(root.visitCount == 1 && near(root.valueSum, 0.3f));
}

void test_backup_records_edge_q_but_not_the_root() {
    BackupNode root{0.0f, 0, 0.0f};
    BackupNode child{0.0f, 0, 0.0f};
    std::vector<BackupNode*> path = {&root, &child};
    MinMaxStats stats;
    backupPath(path, 0.6f, 1.0f, stats);
    // The only edge is root -> child, whose Q is -0.6 in the root's view.
    // With one observation the range is degenerate but must stay finite.
    assert(std::isfinite(stats.normalize(-0.6f)));
    assert(near(child.valueSum, 0.6f));
    assert(near(root.valueSum, -0.6f));
}

void test_node_value_is_zero_before_any_visit() {
    BackupNode node{0.0f, 0, 0.0f};
    assert(near(nodeValue(node), 0.0f));
}

void test_mask_and_renormalize_zeroes_illegal_actions() {
    std::array<float, 9> policy{};
    policy.fill(1.0f / 9.0f);
    std::vector<int> legal = {0, 4, 8};
    std::array<float, 9> masked = MCTS::maskAndRenormalize(policy, legal);

    float sum = 0.0f;
    for (int a = 0; a < 9; ++a) {
        if (a == 0 || a == 4 || a == 8) assert(masked[a] > 0.0f);
        else assert(masked[a] == 0.0f);
        sum += masked[a];
    }
    assert(near(sum, 1.0f));
}

void test_dirichlet_noise_gives_every_legal_action_positive_prior() {
    // A prior of exactly zero must still come out positive, so search can
    // never permanently starve a move the network is wrong about.
    std::array<float, 9> priors{};
    priors.fill(0.0f);
    priors[3] = 1.0f;
    std::vector<int> legal = {0, 3, 6};
    std::mt19937 rng(99);
    MCTS::mixDirichletNoise(priors, legal, rng, 0.3f, 0.25f);
    for (int a : legal) assert(priors[a] > 0.0f);
    assert(priors[1] == 0.0f);   // untouched: not legal
}

void test_search_visits_only_legal_moves_at_the_root() {
    MuZeroNetwork network(1234);
    Board board;
    board = board.applyMove(0);
    board = board.applyMove(4);
    board = board.applyMove(8);

    MCTSConfig config;
    config.numSimulations = 40;
    std::mt19937 rng(7);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);

    for (int a : {0, 4, 8}) assert(result.visitDistribution[a] == 0.0f);
    assert(board.isLegalMove(result.selectedMove));

    float sum = 0.0f;
    for (float v : result.visitDistribution) sum += v;
    assert(near(sum, 1.0f));
}

void test_search_expands_one_node_per_simulation() {
    MuZeroNetwork network(11);
    Board board;
    MCTSConfig config;
    config.numSimulations = 25;
    std::mt19937 rng(3);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);
    assert(result.nodesExpanded == 25);
}

void test_tree_grows_below_a_root_with_only_one_legal_move() {
    // The sharpest test of root-only legality. Fill the board down to a
    // single empty square: the root has exactly one child. If non-root
    // nodes were also masked to legal moves -- or if the tree detected
    // terminal states -- it could not go deeper than that one child. It
    // does, because below the root the search knows neither which actions
    // are legal nor when the game has ended.
    Board board;
    int moves[] = {0, 1, 2, 4, 3, 5, 7, 6};   // eight plies, square 8 open
    for (int m : moves) board = board.applyMove(m);
    assert(board.legalMoves().size() == 1);
    assert(!board.isTerminal());

    MuZeroNetwork network(2);
    MCTSConfig config;
    config.numSimulations = 20;
    std::mt19937 rng(5);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);

    assert(result.selectedMove == 8);
    // Playing square 8 ends the game, so the real position one ply down
    // has no legal moves at all. A rules-aware search could not go below
    // depth 1 here. Depth 2 is reached on the second simulation and is
    // guaranteed, not luck.
    assert(result.maxDepth >= 2);
}

void test_root_value_is_finite_and_bounded() {
    MuZeroNetwork network(19);
    Board board;
    MCTSConfig config;
    config.numSimulations = 30;
    std::mt19937 rng(1);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);
    assert(std::isfinite(result.rootValue));
    assert(result.rootValue >= -2.0f && result.rootValue <= 2.0f);
}

void test_zero_temperature_picks_the_most_visited_move() {
    MuZeroNetwork network(23);
    Board board;
    MCTSConfig config;
    config.numSimulations = 50;
    std::mt19937 rng(9);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);

    float best = 0.0f;
    for (float v : result.visitDistribution) best = std::fmax(best, v);
    assert(near(result.visitDistribution[result.selectedMove], best));
}

void test_positive_temperature_can_pick_something_other_than_the_max() {
    MuZeroNetwork network(29);
    Board board;
    MCTSConfig config;
    config.numSimulations = 30;
    std::mt19937 rng(4);

    std::set<int> chosen;
    for (int trial = 0; trial < 60; ++trial) {
        MCTS mcts(network, config, rng);
        chosen.insert(mcts.run(board, 1.0f).selectedMove);
    }
    assert(chosen.size() > 1);
}

void test_root_noise_changes_the_search() {
    MuZeroNetwork network(37);
    Board board;

    MCTSConfig quiet;
    quiet.numSimulations = 40;
    quiet.addRootNoise = false;

    MCTSConfig noisy = quiet;
    noisy.addRootNoise = true;

    std::mt19937 rngA(2), rngB(2);
    MCTS a(network, quiet, rngA);
    MCTS b(network, noisy, rngB);
    MCTSResult ra = a.run(board, 0.0f);
    MCTSResult rb = b.run(board, 0.0f);

    bool differs = false;
    for (int i = 0; i < 9; ++i) {
        if (!near(ra.visitDistribution[i], rb.visitDistribution[i], 1e-4f)) differs = true;
    }
    assert(differs);
}

int main() {
    test_minmax_stats_passes_through_before_any_update();
    test_minmax_stats_normalizes_observed_range_to_unit_interval();
    test_minmax_stats_handles_a_single_observation();
    test_edge_q_negates_the_child_value();
    test_exploration_term_shrinks_as_a_child_is_visited();
    test_exploration_term_scales_with_prior();
    test_exploration_term_grows_with_parent_visits();
    test_backup_alternates_sign_and_carries_reward();
    test_backup_records_edge_q_but_not_the_root();
    test_node_value_is_zero_before_any_visit();
    test_mask_and_renormalize_zeroes_illegal_actions();
    test_dirichlet_noise_gives_every_legal_action_positive_prior();
    test_search_visits_only_legal_moves_at_the_root();
    test_search_expands_one_node_per_simulation();
    test_tree_grows_below_a_root_with_only_one_legal_move();
    test_root_value_is_finite_and_bounded();
    test_zero_temperature_picks_the_most_visited_move();
    test_positive_temperature_can_pick_something_other_than_the_max();
    test_root_noise_changes_the_search();
    std::printf("all mcts tests passed\n");
    return 0;
}
