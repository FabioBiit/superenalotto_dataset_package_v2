#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>

namespace se::inference {

struct PPCResult {
    Eigen::VectorXd observed_stats;
    Eigen::MatrixXd simulated_stats;
    Eigen::VectorXd p_values_two_sided;
    Eigen::VectorXd p_values_lower;
    Eigen::VectorXd p_values_upper;
};

[[nodiscard]] PPCResult posterior_predictive_check(
    const se::core::DrawSet& observed,
    const Eigen::MatrixXd& omega_posterior_samples,
    int n_simulations_per_sample = 1);

}  // namespace se::inference
