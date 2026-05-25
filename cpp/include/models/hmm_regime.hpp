#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>

namespace se::models {

struct HMMConfig {
    int n_states = 2;
    int max_iter = 200;
    double tol   = 1e-6;
    std::uint64_t seed = 11;
};

struct HMMFit {
    Eigen::VectorXd  pi0;
    Eigen::MatrixXd  transition;
    Eigen::MatrixXd  emission;
    double           log_likelihood{0.0};
    int              n_iterations{0};
    bool             converged{false};
    Eigen::VectorXi  viterbi_path;
};

class HMMRegime {
public:
    explicit HMMRegime(HMMConfig cfg = {});

    [[nodiscard]] HMMFit fit_baum_welch(const Eigen::MatrixXi& observations);

    [[nodiscard]] Eigen::VectorXi viterbi(const Eigen::MatrixXi& observations,
                                            const HMMFit& fit) const;

private:
    HMMConfig cfg_;
};

}  // namespace se::models
