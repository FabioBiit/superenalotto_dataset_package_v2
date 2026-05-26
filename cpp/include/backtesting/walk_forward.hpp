#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace se::backtesting {

struct WalkForwardConfig {
    int initial_train = 800;
    int refit_every   = 100;
};

struct ModelResult {
    std::string     name;
    int             k_params{0};
    int             n_test{0};
    double          log_likelihood{0.0};
    double          aic{0.0};
    double          bic{0.0};
    double          brier_score{0.0};
    double          avg_hits_at_6{0.0};
    double          hits_std{0.0};
    Eigen::VectorXi hits_distribution; // counts of hit_0..hit_6
    Eigen::VectorXi per_draw_hits;     // length n_test
    double          lift_vs_uniform_pct{0.0};
};

struct PermResult {
    double delta_hits_vs_G0{0.0};
    double p_value{1.0};
    bool   robustly_beats_G0{false};
};

class WalkForward {
public:
    explicit WalkForward(WalkForwardConfig cfg = {});

    // Build indicator matrix Y (N x 90), Y(t, i) = 1 iff (i+1) drawn at draw t.
    [[nodiscard]] static Eigen::MatrixXi build_indicator(const se::core::DrawSet& draws);

    // Build regime vector (length N): 0=pre-Flutter, 1=Flutter, 2=4-week.
    [[nodiscard]] static Eigen::VectorXi build_regimes(const se::core::DrawSet& draws);

    // Q-matrices: q[t, i] = P(number (i+1) drawn at draw t | data 0..t-1).
    [[nodiscard]] Eigen::MatrixXd qmat_G0(const se::core::DrawSet& draws) const;
    [[nodiscard]] Eigen::MatrixXd qmat_G2(const se::core::DrawSet& draws, double alpha = 1.0) const;
    [[nodiscard]] Eigen::MatrixXd qmat_G3(const se::core::DrawSet& draws, int lag = 1, double alpha = 1.0) const;
    [[nodiscard]] Eigen::MatrixXd qmat_G4(const se::core::DrawSet& draws) const;
    [[nodiscard]] Eigen::MatrixXd qmat_G5(const se::core::DrawSet& draws, double alpha = 1.0) const;
    [[nodiscard]] Eigen::MatrixXd qmat_G6(const Eigen::MatrixXd& g2,
                                            const Eigen::MatrixXd& g3,
                                            const Eigen::MatrixXd& g4,
                                            std::array<double, 3> w = {0.5, 0.3, 0.2}) const;

    [[nodiscard]] ModelResult evaluate(const std::string& name,
                                         const Eigen::MatrixXd& q,
                                         const se::core::DrawSet& draws,
                                         int k_params) const;

    [[nodiscard]] PermResult paired_permutation_test(const Eigen::VectorXi& hits_a,
                                                       const Eigen::VectorXi& hits_b,
                                                       int n_iter = 4000,
                                                       std::uint64_t seed = 7) const;

    // High-level pipeline: fit and evaluate G0..G6 + run permutation tests.
    struct FullReport {
        std::vector<ModelResult> models;
        std::vector<std::pair<std::string, PermResult>> permutation_tests;
        std::string verdict;
        bool        robust_signal{false};
    };
    [[nodiscard]] FullReport run_all(const se::core::DrawSet& draws) const;

private:
    WalkForwardConfig cfg_;
};

}  // namespace se::backtesting
