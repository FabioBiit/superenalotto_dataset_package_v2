#include "models/wallenius.hpp"

#include "core/prng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace se::models {

Wallenius::Wallenius(int n_balls, int n_pick) : n_balls_(n_balls), n_pick_(n_pick) {
    if (n_balls <= 0 || n_pick <= 0 || n_pick > n_balls)
        throw std::invalid_argument("Wallenius: invalid n_balls/n_pick");
}

double Wallenius::log_pmf(const Eigen::VectorXi& draw_indices, const Eigen::VectorXd& omega) const {
    // Ordered sequential PMF along the sorted draw (canonical ordering).
    // Returns log P(this specific ordering of picks), which differs from the
    // unordered PMF by a fixed constant log(k!) for k distinct picks. That
    // constant is irrelevant for MH-ratio and BIC/AIC ranking across models.
    if (draw_indices.size() != n_pick_)
        throw std::invalid_argument("Wallenius::log_pmf: draw size mismatch");
    if (omega.size() != n_balls_)
        throw std::invalid_argument("Wallenius::log_pmf: omega size mismatch");

    std::vector<int> sorted_idx(n_pick_);
    for (int k = 0; k < n_pick_; ++k) sorted_idx[k] = draw_indices(k) - 1;
    std::sort(sorted_idx.begin(), sorted_idx.end());

    double sum_remaining = omega.sum();
    double ll = 0.0;
    for (int k = 0; k < n_pick_; ++k) {
        const int idx = sorted_idx[k];
        const double w_i = omega(idx);
        if (w_i <= 0.0) return -std::numeric_limits<double>::infinity();
        ll += std::log(w_i) - std::log(std::max(sum_remaining, 1e-300));
        sum_remaining -= w_i;
        if (sum_remaining <= 0.0 && k < n_pick_ - 1) {
            return -std::numeric_limits<double>::infinity();
        }
    }
    return ll;
}

Eigen::VectorXi Wallenius::sample_one(const Eigen::VectorXd& omega, std::uint64_t seed) const {
    if (omega.size() != n_balls_)
        throw std::invalid_argument("Wallenius::sample_one: omega size mismatch");

    auto prng = se::core::make_prng("PCG64", seed);
    Eigen::VectorXi out(n_pick_);
    Eigen::VectorXd w = omega;

    for (int k = 0; k < n_pick_; ++k) {
        const double total = w.sum();
        if (total <= 0.0)
            throw std::runtime_error("Wallenius::sample_one: weights exhausted");

        const double r = prng->next_unit() * total;
        double cum = 0.0;
        int chosen = -1;
        for (int i = 0; i < n_balls_; ++i) {
            cum += w(i);
            if (cum >= r) { chosen = i; break; }
        }
        if (chosen < 0) {
            for (int i = n_balls_ - 1; i >= 0; --i) if (w(i) > 0) { chosen = i; break; }
        }
        out(k) = chosen + 1;
        w(chosen) = 0.0;
    }
    return out;
}

double Wallenius::total_log_likelihood(const se::core::DrawSet& draws,
                                         const Eigen::VectorXd& omega) const {
    double ll = 0.0;
    Eigen::VectorXi idx(n_pick_);
    for (const auto& d : draws) {
        for (int k = 0; k < n_pick_; ++k) idx(k) = d.main[k];
        ll += log_pmf(idx, omega);
    }
    return ll;
}

}  // namespace se::models
