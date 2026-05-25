#include "inference/mcmc_engine.hpp"

#include "core/prng.hpp"

#include <cmath>
#include <stdexcept>

namespace se::inference {

namespace {

double standard_normal(se::core::PRNG& prng) {
    const double u1 = std::max(prng.next_unit(), 1e-300);
    const double u2 = prng.next_unit();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.141592653589793 * u2);
}

}  // namespace

MHResult metropolis_hastings(LogPosterior log_post,
                              const Eigen::VectorXd& init,
                              const MHConfig& cfg) {
    if (!log_post) throw std::invalid_argument("log_post must not be null");
    if (init.size() == 0) throw std::invalid_argument("init must not be empty");
    if (cfg.n_iterations <= 0) throw std::invalid_argument("n_iterations must be > 0");

    const int dim = static_cast<int>(init.size());
    auto prng = se::core::make_prng("PCG64", cfg.seed);

    Eigen::VectorXd current = init;
    double current_lp = log_post(current);

    const int n_kept = std::max(0, cfg.n_iterations - cfg.n_burn_in);
    Eigen::MatrixXd chain(n_kept, dim);
    Eigen::VectorXd lp_chain(n_kept);

    int accepted = 0;
    int kept = 0;
    double step = cfg.step_size;
    int adapt_window_acc = 0;

    for (int iter = 0; iter < cfg.n_iterations; ++iter) {
        Eigen::VectorXd proposal(dim);
        for (int i = 0; i < dim; ++i) {
            proposal(i) = current(i) + step * standard_normal(*prng);
        }
        const double proposal_lp = log_post(proposal);

        const double log_alpha = proposal_lp - current_lp;
        const bool accept = std::log(std::max(prng->next_unit(), 1e-300)) < log_alpha;

        if (accept) {
            current    = proposal;
            current_lp = proposal_lp;
            ++accepted;
            ++adapt_window_acc;
        }

        if (cfg.adapt && iter < cfg.n_burn_in && (iter + 1) % 100 == 0) {
            const double rate = static_cast<double>(adapt_window_acc) / 100.0;
            if (rate > 0.45) step *= 1.10;
            else if (rate < 0.20) step *= 0.90;
            adapt_window_acc = 0;
        }

        if (iter >= cfg.n_burn_in) {
            chain.row(kept)  = current.transpose();
            lp_chain(kept)   = current_lp;
            ++kept;
        }
    }

    MHResult r;
    r.chain           = chain.topRows(kept);
    r.log_post_chain  = lp_chain.head(kept);
    r.acceptance_rate = static_cast<double>(accepted) / cfg.n_iterations;
    return r;
}

}  // namespace se::inference
