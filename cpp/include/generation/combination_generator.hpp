#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

namespace se::generation {

enum class Strategy : std::uint8_t {
    AntiPatternBalanced,
    AntiRecent,
    DelayWeighted,
    FreqWeighted,
    MixedBalance,
    MonteCarloUniform,
};

struct Combination {
    std::array<int, se::core::N_MAIN> main{};
    int    jolly{0};
    int    superstar{0};
    double score{0.0};
    Strategy strategy{Strategy::MonteCarloUniform};
};

struct GeneratorConfig {
    int           n_per_strategy = 8;
    std::uint64_t seed           = 42;
    int           overlap_window = 10;
};

[[nodiscard]] std::vector<Combination> generate_candidates(
    const se::core::DrawSet& history,
    const Eigen::VectorXd& number_weights,
    const GeneratorConfig& cfg = {});

[[nodiscard]] std::string strategy_name(Strategy s);

}  // namespace se::generation
