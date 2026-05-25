#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>

namespace se::models {

class Wallenius {
public:
    explicit Wallenius(int n_balls = se::core::N_MAX, int n_pick = se::core::N_MAIN);

    [[nodiscard]] double log_pmf(const Eigen::VectorXi& draw_indices,
                                  const Eigen::VectorXd& omega) const;

    [[nodiscard]] Eigen::VectorXi sample_one(const Eigen::VectorXd& omega,
                                              std::uint64_t seed) const;

    [[nodiscard]] double total_log_likelihood(const se::core::DrawSet& draws,
                                                const Eigen::VectorXd& omega) const;

private:
    int n_balls_;
    int n_pick_;
};

}  // namespace se::models
