#include "models/dirichlet_mcmc.hpp"

#include "core/prng.hpp"
#include "inference/mcmc_engine.hpp"
#include "models/wallenius.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace se::models {

DirichletMCMC::DirichletMCMC(DirichletMCMCConfig cfg) : cfg_(std::move(cfg)) {}

PosteriorSample DirichletMCMC::fit(const se::core::DrawSet& draws) {
    if (draws.empty()) throw std::invalid_argument("DirichletMCMC::fit: empty draws");
    if (cfg_.n_iterations <= cfg_.n_burn_in)
        throw std::invalid_argument("n_iterations must exceed n_burn_in");

    const int K = se::core::N_MAX;
    Wallenius wal(K, se::core::N_MAIN);

    // Parameterize on log-omega for unconstrained sampling. omega = exp(eta).
    // Identifiability: fix sum log-omega = 0 implicitly by tracking only K-1 params,
    // but for MH simplicity keep all K and add a soft sum-to-zero prior via
    // anchoring the geometric mean to 1 (subtract mean from chain at output).
    auto log_post = [&](const Eigen::VectorXd& eta) -> double {
        Eigen::VectorXd omega(K);
        for (int i = 0; i < K; ++i) omega(i) = std::exp(eta(i));
        const double ll = wal.total_log_likelihood(draws, omega);
        // Symmetric Dirichlet prior on normalized omega:
        // log p(omega) ~ (alpha - 1) * sum log(omega_i / sum_omega)
        const double sum_omega = omega.sum();
        double log_prior = 0.0;
        for (int i = 0; i < K; ++i) {
            log_prior += (cfg_.prior_alpha - 1.0) *
                         (eta(i) - std::log(std::max(sum_omega, 1e-300)));
        }
        return ll + log_prior;
    };

    se::inference::MHConfig mh_cfg;
    mh_cfg.n_iterations = cfg_.n_iterations;
    mh_cfg.n_burn_in    = cfg_.n_burn_in;
    mh_cfg.step_size    = 0.05;
    mh_cfg.adapt        = cfg_.adapt_proposal;
    mh_cfg.seed         = cfg_.seed;

    const Eigen::VectorXd init = Eigen::VectorXd::Zero(K);
    auto mh_result = se::inference::metropolis_hastings(log_post, init, mh_cfg);

    const int n_post = static_cast<int>(mh_result.chain.rows());
    const int n_kept = (n_post + cfg_.n_thin - 1) / cfg_.n_thin;

    PosteriorSample s;
    s.omega_chain          = Eigen::MatrixXd(n_kept, K);
    s.log_likelihood_chain = Eigen::VectorXd(n_kept);
    int row = 0;
    for (int i = 0; i < n_post; i += cfg_.n_thin) {
        Eigen::VectorXd eta = mh_result.chain.row(i);
        const double mean_eta = eta.mean();
        eta.array() -= mean_eta;                       // center for identifiability
        for (int j = 0; j < K; ++j) s.omega_chain(row, j) = std::exp(eta(j));
        s.log_likelihood_chain(row) = mh_result.log_post_chain(i);
        ++row;
    }
    s.omega_chain.conservativeResize(row, K);
    s.log_likelihood_chain.conservativeResize(row);
    s.acceptance_rate = mh_result.acceptance_rate;
    s.n_effective     = row;
    return s;
}

Eigen::VectorXd DirichletMCMC::posterior_mean_omega(const PosteriorSample& s) const {
    if (s.omega_chain.rows() == 0)
        throw std::invalid_argument("posterior_mean_omega: empty chain");
    return s.omega_chain.colwise().mean();
}

}  // namespace se::models
