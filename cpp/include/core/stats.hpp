#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>
#include <span>

namespace se::core::stats {

struct TestResult {
    double statistic{0.0};
    double p_value{1.0};
    int    df{0};
    bool   reject_h0{false};
};

[[nodiscard]] TestResult chi_square_uniform(std::span<const int> counts, int expected_buckets);
[[nodiscard]] TestResult ks_test_uniform(std::span<const double> samples);
[[nodiscard]] TestResult runs_test(std::span<const int> binary_sequence);
[[nodiscard]] TestResult ljung_box(std::span<const double> series, int max_lag);

[[nodiscard]] double shannon_entropy(std::span<const double> probs);
[[nodiscard]] double kl_divergence(std::span<const double> p, std::span<const double> q);
[[nodiscard]] double js_divergence(std::span<const double> p, std::span<const double> q);

[[nodiscard]] Eigen::VectorXd empirical_frequency(const DrawSet& draws);
[[nodiscard]] Eigen::VectorXd delays(const DrawSet& draws);
[[nodiscard]] Eigen::MatrixXd cooccurrence_matrix(const DrawSet& draws);

}  // namespace se::core::stats
