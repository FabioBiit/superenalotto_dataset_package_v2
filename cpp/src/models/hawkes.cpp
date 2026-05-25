#include "models/hawkes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace se::models {

namespace {

double hawkes_log_likelihood(const Eigen::VectorXd& t,
                              double mu, double alpha, double beta) {
    if (mu <= 0.0 || alpha < 0.0 || beta <= 0.0) return -std::numeric_limits<double>::infinity();
    const int n = static_cast<int>(t.size());
    if (n == 0) return 0.0;
    const double T = t(n - 1);

    // Recursive intensity (Ogata 1981): A(i) = exp(-beta*(t_i - t_{i-1})) * (1 + A(i-1))
    // lambda(t_i) = mu + alpha * beta * A(i)  (A(0) = 0)
    double A = 0.0;
    double ll = 0.0;
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            const double dt = t(i) - t(i - 1);
            A = std::exp(-beta * dt) * (1.0 + A);
        }
        const double lam = mu + alpha * beta * A;
        if (lam <= 0.0) return -std::numeric_limits<double>::infinity();
        ll += std::log(lam);
    }
    // Integral term: int_0^T lambda(s) ds = mu*T + alpha * sum_i (1 - exp(-beta*(T - t_i)))
    double integral = mu * T;
    for (int i = 0; i < n; ++i) {
        integral += alpha * (1.0 - std::exp(-beta * (T - t(i))));
    }
    ll -= integral;
    return ll;
}

}  // namespace

HawkesFit Hawkes::fit(const Eigen::VectorXd& event_times, int max_iter) {
    if (event_times.size() < 5)
        throw std::invalid_argument("Hawkes::fit: need at least 5 event times");

    // Simple grid-refinement search over (mu, alpha, beta).
    // Starts from a rate-based initial guess.
    const double T = event_times(event_times.size() - 1);
    const double rate = static_cast<double>(event_times.size()) / std::max(T, 1.0);

    HawkesFit best;
    best.params.mu    = rate;
    best.params.alpha = 0.0;
    best.params.beta  = 1.0;
    best.log_likelihood = hawkes_log_likelihood(event_times, rate, 0.0, 1.0);

    double mu_lo = rate * 0.1, mu_hi = rate * 2.0;
    double a_lo  = 0.0,        a_hi  = 0.99;
    double b_lo  = 0.05,       b_hi  = 5.0;

    for (int round = 0; round < max_iter; ++round) {
        const int grid = 7;
        for (int im = 0; im < grid; ++im) {
            const double mu = mu_lo + (mu_hi - mu_lo) * im / (grid - 1);
            for (int ia = 0; ia < grid; ++ia) {
                const double a = a_lo + (a_hi - a_lo) * ia / (grid - 1);
                for (int ib = 0; ib < grid; ++ib) {
                    const double b = b_lo + (b_hi - b_lo) * ib / (grid - 1);
                    const double ll = hawkes_log_likelihood(event_times, mu, a, b);
                    if (ll > best.log_likelihood) {
                        best.log_likelihood = ll;
                        best.params.mu = mu;
                        best.params.alpha = a;
                        best.params.beta  = b;
                    }
                }
            }
        }
        const double mu_w = (mu_hi - mu_lo) * 0.3;
        const double a_w  = (a_hi  - a_lo)  * 0.3;
        const double b_w  = (b_hi  - b_lo)  * 0.3;
        mu_lo = std::max(1e-6, best.params.mu - mu_w);
        mu_hi = best.params.mu + mu_w;
        a_lo  = std::max(0.0,   best.params.alpha - a_w);
        a_hi  = std::min(0.99,  best.params.alpha + a_w);
        b_lo  = std::max(1e-3,  best.params.beta - b_w);
        b_hi  = best.params.beta + b_w;
        if (mu_w + a_w + b_w < 1e-5) break;
    }
    best.converged = true;
    return best;
}

double Hawkes::intensity_at(double t, const Eigen::VectorXd& past_events, const HawkesParams& p) {
    double s = p.mu;
    for (int i = 0; i < past_events.size(); ++i) {
        const double dt = t - past_events(i);
        if (dt > 0.0) s += p.alpha * p.beta * std::exp(-p.beta * dt);
    }
    return s;
}

}  // namespace se::models
