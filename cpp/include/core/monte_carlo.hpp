#pragma once

#include "core/prng.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <span>

namespace se::core {

struct MCConfig {
    std::uint64_t n_simulations = 1'000'000;
    std::uint64_t seed = 42;
    int           n_threads = 0;
    bool          use_cuda = false;
};

[[nodiscard]] Eigen::VectorXd simulate_uniform_frequency(int n_draws, const MCConfig& cfg = {});

[[nodiscard]] Eigen::VectorXd simulate_weighted_frequency(
    std::span<const double> weights, int n_draws, const MCConfig& cfg = {});

[[nodiscard]] Eigen::VectorXd permutation_pvalues(
    std::span<const int> observed_counts,
    std::span<const double> baseline_probs,
    int n_permutations,
    const MCConfig& cfg = {});

}  // namespace se::core
