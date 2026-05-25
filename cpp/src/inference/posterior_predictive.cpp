#include "inference/posterior_predictive.hpp"

#include "core/prng.hpp"
#include "models/wallenius.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace se::inference {

namespace {

constexpr int N_STATS = 5;
// 0: chi2 vs uniform     2: mean evens (parity 3:3 deviation)
// 1: mean sum            3: mean distinct decades
//                        4: mean consecutive-pair count

Eigen::VectorXd compute_stats(const std::vector<std::array<int, se::core::N_MAIN>>& draws_list) {
    Eigen::VectorXd s = Eigen::VectorXd::Zero(N_STATS);
    if (draws_list.empty()) return s;

    const int n = static_cast<int>(draws_list.size());
    std::array<int, se::core::N_MAX> freq{};
    double sum_sum = 0.0, sum_even = 0.0, sum_dec = 0.0, sum_consec = 0.0;

    for (const auto& d : draws_list) {
        std::array<int, se::core::N_MAIN> sorted = d;
        std::sort(sorted.begin(), sorted.end());
        bool seen_dec[9] = {false};
        int sum = 0, evens = 0, consec = 0;
        for (int k = 0; k < se::core::N_MAIN; ++k) {
            ++freq[sorted[k] - 1];
            sum += sorted[k];
            evens += (sorted[k] % 2 == 0) ? 1 : 0;
            const int dec = (sorted[k] - 1) / 10;
            if (dec >= 0 && dec < 9) seen_dec[dec] = true;
            if (k > 0 && sorted[k] == sorted[k - 1] + 1) ++consec;
        }
        int dec_count = 0;
        for (bool s_ : seen_dec) if (s_) ++dec_count;

        sum_sum    += sum;
        sum_even   += evens;
        sum_dec    += dec_count;
        sum_consec += consec;
    }

    const double exp_per = n * se::core::N_MAIN / static_cast<double>(se::core::N_MAX);
    double chi2 = 0.0;
    for (int v : freq) {
        const double diff = v - exp_per;
        chi2 += (diff * diff) / std::max(exp_per, 1e-12);
    }

    s(0) = chi2;
    s(1) = sum_sum    / n;
    s(2) = sum_even   / n;
    s(3) = sum_dec    / n;
    s(4) = sum_consec / n;
    return s;
}

}  // namespace

PPCResult posterior_predictive_check(
    const se::core::DrawSet& observed,
    const Eigen::MatrixXd& omega_posterior_samples,
    int n_simulations_per_sample) {

    if (observed.empty()) throw std::invalid_argument("PPC: observed is empty");
    if (n_simulations_per_sample <= 0) throw std::invalid_argument("PPC: n_sims must be > 0");

    const int n_draws = static_cast<int>(observed.size());
    const int S       = static_cast<int>(omega_posterior_samples.rows());
    if (S == 0) throw std::invalid_argument("PPC: empty posterior chain");
    const int K       = static_cast<int>(omega_posterior_samples.cols());

    std::vector<std::array<int, se::core::N_MAIN>> obs_list;
    obs_list.reserve(n_draws);
    for (const auto& d : observed) obs_list.push_back(d.main);
    const Eigen::VectorXd obs_stats = compute_stats(obs_list);

    const int total_sims = S * n_simulations_per_sample;
    Eigen::MatrixXd sim_stats(total_sims, N_STATS);

    se::models::Wallenius wal(K, se::core::N_MAIN);
    auto prng = se::core::make_prng("PCG64", 0xC0FFEE);

    for (int s = 0; s < S; ++s) {
        Eigen::VectorXd omega = omega_posterior_samples.row(s).transpose();
        if ((omega.array() <= 0).any()) omega = omega.array().max(1e-9);

        for (int sim = 0; sim < n_simulations_per_sample; ++sim) {
            std::vector<std::array<int, se::core::N_MAIN>> sim_draws;
            sim_draws.reserve(n_draws);
            for (int d = 0; d < n_draws; ++d) {
                const std::uint64_t seed = prng->next_u64();
                const auto picks = wal.sample_one(omega, seed);
                std::array<int, se::core::N_MAIN> arr{};
                for (int k = 0; k < se::core::N_MAIN; ++k) arr[k] = picks(k);
                sim_draws.push_back(arr);
            }
            sim_stats.row(s * n_simulations_per_sample + sim) =
                compute_stats(sim_draws).transpose();
        }
    }

    PPCResult r;
    r.observed_stats     = obs_stats;
    r.simulated_stats    = sim_stats;
    r.p_values_lower     = Eigen::VectorXd::Zero(N_STATS);
    r.p_values_upper     = Eigen::VectorXd::Zero(N_STATS);
    r.p_values_two_sided = Eigen::VectorXd::Zero(N_STATS);
    for (int i = 0; i < N_STATS; ++i) {
        int n_below = 0, n_above = 0;
        for (int j = 0; j < total_sims; ++j) {
            if (sim_stats(j, i) <= obs_stats(i)) ++n_below;
            if (sim_stats(j, i) >= obs_stats(i)) ++n_above;
        }
        r.p_values_lower(i) = static_cast<double>(n_below) / total_sims;
        r.p_values_upper(i) = static_cast<double>(n_above) / total_sims;
        r.p_values_two_sided(i) = 2.0 * std::min(r.p_values_lower(i), r.p_values_upper(i));
    }
    return r;
}

}  // namespace se::inference
