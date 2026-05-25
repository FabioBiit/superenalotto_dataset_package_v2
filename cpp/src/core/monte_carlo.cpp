#include "core/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef SE_HAS_OPENMP
#include <omp.h>
#endif

namespace se::core {

namespace {

void sample_six_without_replacement_uniform(PRNG& prng, std::array<int, N_MAIN>& out) {
    std::array<int, N_MAX> deck;
    std::iota(deck.begin(), deck.end(), 1);
    for (int i = 0; i < N_MAIN; ++i) {
        const std::uint64_t r = prng.next_u64();
        const int j = i + static_cast<int>(r % static_cast<std::uint64_t>(N_MAX - i));
        std::swap(deck[i], deck[j]);
        out[i] = deck[i];
    }
}

void sample_six_weighted_no_replacement(PRNG& prng,
                                          std::span<const double> weights,
                                          std::array<int, N_MAIN>& out) {
    std::array<double, N_MAX> w;
    std::array<bool,   N_MAX> picked{};
    for (int i = 0; i < N_MAX; ++i) w[i] = weights[i];

    for (int k = 0; k < N_MAIN; ++k) {
        double total = 0.0;
        for (int i = 0; i < N_MAX; ++i) if (!picked[i]) total += w[i];
        if (total <= 0.0) {
            for (int i = 0; i < N_MAX; ++i) if (!picked[i]) { out[k] = i + 1; picked[i] = true; break; }
            continue;
        }
        const double r = prng.next_unit() * total;
        double cum = 0.0;
        int chosen = -1;
        for (int i = 0; i < N_MAX; ++i) {
            if (picked[i]) continue;
            cum += w[i];
            if (cum >= r) { chosen = i; break; }
        }
        if (chosen < 0) {
            for (int i = N_MAX - 1; i >= 0; --i) if (!picked[i]) { chosen = i; break; }
        }
        out[k] = chosen + 1;
        picked[chosen] = true;
    }
}

}  // namespace

Eigen::VectorXd simulate_uniform_frequency(int n_draws, const MCConfig& cfg) {
    if (n_draws <= 0) throw std::invalid_argument("n_draws must be > 0");

    Eigen::VectorXd freq = Eigen::VectorXd::Zero(N_MAX);

#ifdef SE_HAS_OPENMP
    const int nthreads = (cfg.n_threads > 0) ? cfg.n_threads : omp_get_max_threads();
    std::vector<Eigen::VectorXd> per_thread(nthreads, Eigen::VectorXd::Zero(N_MAX));

    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto prng = make_prng("PCG64", cfg.seed ^ (0x9E3779B97F4A7C15ULL * (tid + 1)));
        std::array<int, N_MAIN> picks{};

        #pragma omp for schedule(static)
        for (int d = 0; d < n_draws; ++d) {
            sample_six_without_replacement_uniform(*prng, picks);
            for (int v : picks) per_thread[tid](v - 1) += 1.0;
        }
    }
    for (const auto& t : per_thread) freq += t;
#else
    auto prng = make_prng("PCG64", cfg.seed);
    std::array<int, N_MAIN> picks{};
    for (int d = 0; d < n_draws; ++d) {
        sample_six_without_replacement_uniform(*prng, picks);
        for (int v : picks) freq(v - 1) += 1.0;
    }
#endif

    const double denom = static_cast<double>(n_draws) * N_MAIN;
    if (denom > 0) freq /= denom;
    return freq;
}

Eigen::VectorXd simulate_weighted_frequency(std::span<const double> weights,
                                              int n_draws,
                                              const MCConfig& cfg) {
    if (weights.size() != static_cast<std::size_t>(N_MAX))
        throw std::invalid_argument("weights must have length 90");
    if (n_draws <= 0) throw std::invalid_argument("n_draws must be > 0");

    Eigen::VectorXd freq = Eigen::VectorXd::Zero(N_MAX);

#ifdef SE_HAS_OPENMP
    const int nthreads = (cfg.n_threads > 0) ? cfg.n_threads : omp_get_max_threads();
    std::vector<Eigen::VectorXd> per_thread(nthreads, Eigen::VectorXd::Zero(N_MAX));

    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto prng = make_prng("PCG64", cfg.seed ^ (0xC2B2AE3D27D4EB4FULL * (tid + 1)));
        std::array<int, N_MAIN> picks{};

        #pragma omp for schedule(static)
        for (int d = 0; d < n_draws; ++d) {
            sample_six_weighted_no_replacement(*prng, weights, picks);
            for (int v : picks) per_thread[tid](v - 1) += 1.0;
        }
    }
    for (const auto& t : per_thread) freq += t;
#else
    auto prng = make_prng("PCG64", cfg.seed);
    std::array<int, N_MAIN> picks{};
    for (int d = 0; d < n_draws; ++d) {
        sample_six_weighted_no_replacement(*prng, weights, picks);
        for (int v : picks) freq(v - 1) += 1.0;
    }
#endif

    const double denom = static_cast<double>(n_draws) * N_MAIN;
    if (denom > 0) freq /= denom;
    return freq;
}

Eigen::VectorXd permutation_pvalues(std::span<const int> observed_counts,
                                      std::span<const double> baseline_probs,
                                      int n_permutations,
                                      const MCConfig& cfg) {
    if (observed_counts.size() != static_cast<std::size_t>(N_MAX) ||
        baseline_probs.size()  != static_cast<std::size_t>(N_MAX))
        throw std::invalid_argument("vectors must have length 90");
    if (n_permutations <= 0) throw std::invalid_argument("n_permutations must be > 0");

    const int total = std::accumulate(observed_counts.begin(), observed_counts.end(), 0);
    const int n_draws = total / N_MAIN;
    if (n_draws <= 0) throw std::invalid_argument("observed_counts imply zero draws");

    Eigen::VectorXd obs_dev(N_MAX);
    Eigen::VectorXd exp_count(N_MAX);
    for (int i = 0; i < N_MAX; ++i) {
        exp_count(i) = n_draws * N_MAIN * baseline_probs[i];
        obs_dev(i)   = std::abs(observed_counts[i] - exp_count(i));
    }

    Eigen::VectorXd p_vals = Eigen::VectorXd::Zero(N_MAX);

#ifdef SE_HAS_OPENMP
    const int nthreads = (cfg.n_threads > 0) ? cfg.n_threads : omp_get_max_threads();
    std::vector<Eigen::VectorXd> per_thread(nthreads, Eigen::VectorXd::Zero(N_MAX));

    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto prng = make_prng("PCG64", cfg.seed ^ (0xDA3E39CB94B95BDBULL * (tid + 1)));
        std::array<int, N_MAIN> picks{};
        std::vector<int> sim_counts(N_MAX, 0);

        #pragma omp for schedule(static)
        for (int sim = 0; sim < n_permutations; ++sim) {
            std::fill(sim_counts.begin(), sim_counts.end(), 0);
            for (int d = 0; d < n_draws; ++d) {
                sample_six_weighted_no_replacement(*prng, baseline_probs, picks);
                for (int v : picks) ++sim_counts[v - 1];
            }
            for (int i = 0; i < N_MAX; ++i) {
                const double dev = std::abs(sim_counts[i] - exp_count(i));
                if (dev >= obs_dev(i)) per_thread[tid](i) += 1.0;
            }
        }
    }
    for (const auto& t : per_thread) p_vals += t;
#else
    auto prng = make_prng("PCG64", cfg.seed);
    std::array<int, N_MAIN> picks{};
    std::vector<int> sim_counts(N_MAX, 0);
    for (int sim = 0; sim < n_permutations; ++sim) {
        std::fill(sim_counts.begin(), sim_counts.end(), 0);
        for (int d = 0; d < n_draws; ++d) {
            sample_six_weighted_no_replacement(*prng, baseline_probs, picks);
            for (int v : picks) ++sim_counts[v - 1];
        }
        for (int i = 0; i < N_MAX; ++i) {
            const double dev = std::abs(sim_counts[i] - exp_count(i));
            if (dev >= obs_dev(i)) p_vals(i) += 1.0;
        }
    }
#endif

    p_vals /= static_cast<double>(n_permutations);
    return p_vals;
}

}  // namespace se::core
