#include "models/hmm_regime.hpp"

#include "core/prng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace se::models {

namespace {

int one_hot_index(const Eigen::MatrixXi& obs, int t) {
    for (int d = 0; d < obs.cols(); ++d)
        if (obs(t, d) == 1) return d;
    return -1;
}

}  // namespace

HMMRegime::HMMRegime(HMMConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.n_states < 2) throw std::invalid_argument("HMM requires n_states >= 2");
}

HMMFit HMMRegime::fit_baum_welch(const Eigen::MatrixXi& observations) {
    const int T = static_cast<int>(observations.rows());
    const int D = static_cast<int>(observations.cols());
    const int K = cfg_.n_states;
    if (T < 2 || D < 1) throw std::invalid_argument("HMM: observations too small");

    auto prng = se::core::make_prng("PCG64", cfg_.seed);

    Eigen::VectorXd pi = Eigen::VectorXd::Constant(K, 1.0 / K);
    Eigen::MatrixXd A  = Eigen::MatrixXd::Constant(K, K, 1.0 / K);
    Eigen::MatrixXd B  = Eigen::MatrixXd::Constant(K, D, 1.0 / D);

    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < D; ++d) B(k, d) += 0.1 * prng->next_unit();
        B.row(k) /= B.row(k).sum();
        for (int kp = 0; kp < K; ++kp) A(k, kp) += 0.05 * prng->next_unit();
        A.row(k) /= A.row(k).sum();
    }

    Eigen::MatrixXd alpha(T, K);
    Eigen::MatrixXd beta(T, K);
    Eigen::MatrixXd gamma(T, K);
    Eigen::MatrixXd xi_sum(K, K);
    Eigen::VectorXd c(T);

    double prev_ll = -std::numeric_limits<double>::infinity();
    int iter = 0;
    bool converged = false;

    for (iter = 0; iter < cfg_.max_iter; ++iter) {
        // Forward
        const int o0 = one_hot_index(observations, 0);
        for (int k = 0; k < K; ++k)
            alpha(0, k) = pi(k) * (o0 >= 0 ? B(k, o0) : 1.0);
        c(0) = std::max(alpha.row(0).sum(), 1e-300);
        alpha.row(0) /= c(0);

        for (int t = 1; t < T; ++t) {
            const int ot = one_hot_index(observations, t);
            for (int k = 0; k < K; ++k) {
                double s = 0.0;
                for (int kp = 0; kp < K; ++kp) s += alpha(t - 1, kp) * A(kp, k);
                alpha(t, k) = s * (ot >= 0 ? B(k, ot) : 1.0);
            }
            c(t) = std::max(alpha.row(t).sum(), 1e-300);
            alpha.row(t) /= c(t);
        }

        double ll = 0.0;
        for (int t = 0; t < T; ++t) ll += std::log(c(t));

        // Backward
        beta.row(T - 1).setConstant(1.0 / c(T - 1));
        for (int t = T - 2; t >= 0; --t) {
            const int otn = one_hot_index(observations, t + 1);
            for (int k = 0; k < K; ++k) {
                double s = 0.0;
                for (int kn = 0; kn < K; ++kn)
                    s += A(k, kn) * (otn >= 0 ? B(kn, otn) : 1.0) * beta(t + 1, kn);
                beta(t, k) = s;
            }
            beta.row(t) /= c(t);
        }

        // Gamma
        for (int t = 0; t < T; ++t) {
            const double row_sum = std::max((alpha.row(t).array() * beta.row(t).array()).sum(), 1e-300);
            for (int k = 0; k < K; ++k) gamma(t, k) = alpha(t, k) * beta(t, k) / row_sum;
        }

        // Xi sum
        xi_sum.setZero();
        for (int t = 0; t < T - 1; ++t) {
            const int otn = one_hot_index(observations, t + 1);
            Eigen::MatrixXd xi_t(K, K);
            double denom = 0.0;
            for (int k = 0; k < K; ++k) {
                for (int kn = 0; kn < K; ++kn) {
                    xi_t(k, kn) = alpha(t, k) * A(k, kn) *
                                   (otn >= 0 ? B(kn, otn) : 1.0) * beta(t + 1, kn);
                    denom += xi_t(k, kn);
                }
            }
            if (denom > 0) xi_t /= denom;
            xi_sum += xi_t;
        }

        // M-step
        pi = gamma.row(0).transpose();
        for (int k = 0; k < K; ++k) {
            const double row_sum = xi_sum.row(k).sum();
            if (row_sum > 0) A.row(k) = xi_sum.row(k) / row_sum;
        }
        for (int k = 0; k < K; ++k) {
            Eigen::VectorXd b_new = Eigen::VectorXd::Zero(D);
            double tot = 0.0;
            for (int t = 0; t < T; ++t) {
                tot += gamma(t, k);
                const int ot = one_hot_index(observations, t);
                if (ot >= 0) b_new(ot) += gamma(t, k);
            }
            if (tot > 0) {
                B.row(k) = (b_new / tot).transpose();
                for (int d = 0; d < D; ++d) B(k, d) = std::max(B(k, d), 1e-12);
                B.row(k) /= B.row(k).sum();
            }
        }

        if (std::abs(ll - prev_ll) < cfg_.tol) {
            converged = true;
            prev_ll = ll;
            ++iter;
            break;
        }
        prev_ll = ll;
    }

    HMMFit fit;
    fit.pi0            = pi;
    fit.transition     = A;
    fit.emission       = B;
    fit.log_likelihood = prev_ll;
    fit.n_iterations   = iter;
    fit.converged      = converged;
    fit.viterbi_path   = viterbi(observations, fit);
    return fit;
}

Eigen::VectorXi HMMRegime::viterbi(const Eigen::MatrixXi& observations, const HMMFit& fit) const {
    const int T = static_cast<int>(observations.rows());
    const int K = static_cast<int>(fit.pi0.size());
    if (T == 0) return Eigen::VectorXi();

    Eigen::MatrixXd delta(T, K);
    Eigen::MatrixXi psi(T, K);

    const int o0 = one_hot_index(observations, 0);
    for (int k = 0; k < K; ++k) {
        const double emit = (o0 >= 0 ? fit.emission(k, o0) : 1.0);
        delta(0, k) = std::log(std::max(fit.pi0(k), 1e-300)) + std::log(std::max(emit, 1e-300));
        psi(0, k)   = 0;
    }
    for (int t = 1; t < T; ++t) {
        const int ot = one_hot_index(observations, t);
        for (int k = 0; k < K; ++k) {
            double best = -std::numeric_limits<double>::infinity();
            int    arg  = 0;
            for (int kp = 0; kp < K; ++kp) {
                const double v = delta(t - 1, kp) +
                                  std::log(std::max(fit.transition(kp, k), 1e-300));
                if (v > best) { best = v; arg = kp; }
            }
            const double emit = (ot >= 0 ? fit.emission(k, ot) : 1.0);
            delta(t, k) = best + std::log(std::max(emit, 1e-300));
            psi(t, k)   = arg;
        }
    }
    Eigen::VectorXi path(T);
    double best = -std::numeric_limits<double>::infinity();
    int arg = 0;
    for (int k = 0; k < K; ++k) if (delta(T - 1, k) > best) { best = delta(T - 1, k); arg = k; }
    path(T - 1) = arg;
    for (int t = T - 2; t >= 0; --t) path(t) = psi(t + 1, path(t + 1));
    return path;
}

}  // namespace se::models
