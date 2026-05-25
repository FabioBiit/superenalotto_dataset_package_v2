#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>

namespace se::models {

struct DirichletMCMCConfig {
    int    n_iterations = 50'000;
    int    n_burn_in    = 10'000;
    int    n_thin       = 5;
    double prior_alpha  = 1.0;
    std::uint64_t seed  = 7;
    bool   adapt_proposal = true;
};

struct PosteriorSample {
    Eigen::MatrixXd omega_chain;
    Eigen::VectorXd log_likelihood_chain;
    double acceptance_rate{0.0};
    int    n_effective{0};
};

class DirichletMCMC {
public:
    explicit DirichletMCMC(DirichletMCMCConfig cfg = {});

    [[nodiscard]] PosteriorSample fit(const se::core::DrawSet& draws);

    [[nodiscard]] Eigen::VectorXd posterior_mean_omega(const PosteriorSample& s) const;

private:
    DirichletMCMCConfig cfg_;
};

}  // namespace se::models
