#pragma once

#include "core/prng.hpp"
#include "core/stats.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace se::models {

struct PRNGFingerprintResult {
    std::string prng_name;
    se::core::stats::TestResult chi_square;
    se::core::stats::TestResult ks_test;
    se::core::stats::TestResult runs_test;
    double js_divergence_vs_observed{0.0};
    double overall_score{0.0};
};

class PRNGFingerprint {
public:
    [[nodiscard]] static std::vector<PRNGFingerprintResult> compare_against_observed(
        const se::core::DrawSet& observed,
        const std::vector<std::string>& prng_names = {"PCG64", "Philox", "ChaCha20"},
        std::uint64_t seed = 42);
};

}  // namespace se::models
