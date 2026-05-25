#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "core/monte_carlo.hpp"

using namespace se::core;
using Catch::Matchers::WithinAbs;

TEST_CASE("simulate_uniform_frequency: long-run sums to 1", "[mc]") {
    MCConfig cfg;
    cfg.seed = 42;
    cfg.n_threads = 1;
    auto freq = simulate_uniform_frequency(50'000, cfg);
    REQUIRE(freq.size() == N_MAX);
    REQUIRE_THAT(freq.sum(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("simulate_uniform_frequency: approximately uniform", "[mc]") {
    MCConfig cfg;
    cfg.seed = 7;
    cfg.n_threads = 1;
    auto freq = simulate_uniform_frequency(200'000, cfg);
    const double expected = 1.0 / N_MAX;
    for (int i = 0; i < N_MAX; ++i)
        REQUIRE_THAT(freq(i), WithinAbs(expected, 0.005));
}

TEST_CASE("simulate_weighted_frequency: heavy weight wins", "[mc]") {
    std::vector<double> weights(N_MAX, 1.0);
    weights[0] = 100.0;  // ball 1 has 100x weight
    MCConfig cfg;
    cfg.seed = 11;
    cfg.n_threads = 1;
    auto freq = simulate_weighted_frequency(weights, 10'000, cfg);
    REQUIRE(freq(0) > 1.0 / N_MAX * 5.0);  // ball 1 vastly overrepresented
}

TEST_CASE("simulate_weighted_frequency: validates input size", "[mc]") {
    std::vector<double> wrong_size(89, 1.0);
    REQUIRE_THROWS(simulate_weighted_frequency(wrong_size, 100));
}
