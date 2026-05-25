#include <catch2/catch_test_macros.hpp>
#include "generation/combination_generator.hpp"
#include "generation/mmr_selector.hpp"

using namespace se::generation;

TEST_CASE("jaccard_distance basic", "[gen]") {
    Combination a; a.main = {1, 2, 3, 4, 5, 6};
    Combination b; b.main = {1, 2, 3, 4, 5, 6};
    Combination c; c.main = {7, 8, 9, 10, 11, 12};

    REQUIRE(jaccard_distance(a, b) == 0.0);
    REQUIRE(jaccard_distance(a, c) == 1.0);
}

TEST_CASE("jaccard_distance partial overlap", "[gen]") {
    Combination a; a.main = {1, 2, 3, 4, 5, 6};
    Combination b; b.main = {1, 2, 3, 10, 20, 30};

    // intersection = 3, union = 9 => distance = 1 - 3/9 = 6/9
    REQUIRE(std::abs(jaccard_distance(a, b) - 2.0 / 3.0) < 1e-9);
}

TEST_CASE("strategy_name returns correct labels", "[gen]") {
    REQUIRE(strategy_name(Strategy::AntiPatternBalanced) == "anti_pattern_balanced");
    REQUIRE(strategy_name(Strategy::AntiRecent)          == "anti_recent");
    REQUIRE(strategy_name(Strategy::DelayWeighted)       == "delay_weighted");
    REQUIRE(strategy_name(Strategy::FreqWeighted)        == "freq_weighted");
    REQUIRE(strategy_name(Strategy::MixedBalance)        == "mixed_balance");
    REQUIRE(strategy_name(Strategy::MonteCarloUniform)   == "monte_carlo_uniform");
}

TEST_CASE("select_mmr respects k_final", "[gen]") {
    std::vector<Combination> candidates;
    for (int i = 0; i < 40; ++i) {
        Combination c;
        c.main = {1 + i % 10, 11 + i % 10, 21 + i % 10, 31 + i % 10, 41 + i % 10, 51 + i % 10};
        c.score = static_cast<double>(i);
        candidates.push_back(c);
    }
    MMRConfig cfg;
    cfg.k_final = 10;
    cfg.lambda  = 0.5;
    auto sel = select_mmr(candidates, cfg);
    REQUIRE(sel.size() == 10);
}

TEST_CASE("select_mmr rejects invalid lambda", "[gen]") {
    Combination c; c.main = {1, 2, 3, 4, 5, 6};
    std::vector<Combination> v{c};
    MMRConfig cfg; cfg.lambda = -0.1;
    REQUIRE_THROWS(select_mmr(v, cfg));
    cfg.lambda = 1.5;
    REQUIRE_THROWS(select_mmr(v, cfg));
}
