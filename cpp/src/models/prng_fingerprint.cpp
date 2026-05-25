#include "models/prng_fingerprint.hpp"

#include "core/monte_carlo.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace se::models {

namespace {

Eigen::VectorXd uniform_probs() {
    return Eigen::VectorXd::Constant(se::core::N_MAX, 1.0 / se::core::N_MAX);
}

std::vector<int> draw_counts(const se::core::DrawSet& draws) {
    std::vector<int> c(se::core::N_MAX, 0);
    for (const auto& d : draws) for (int n : d.main) ++c[n - 1];
    return c;
}

double overall_score(const PRNGFingerprintResult& r) {
    // Combined p-value score: closer to observed = higher score.
    // Each p-value contributes log10(p) where p large (close to 1) = indistinguishable.
    double s = 0.0;
    s += std::log10(std::max(r.chi_square.p_value, 1e-12));
    s += std::log10(std::max(r.ks_test.p_value,    1e-12));
    s += std::log10(std::max(r.runs_test.p_value,  1e-12));
    s -= 10.0 * r.js_divergence_vs_observed;   // smaller JS = better
    return s;
}

}  // namespace

std::vector<PRNGFingerprintResult> PRNGFingerprint::compare_against_observed(
    const se::core::DrawSet& observed,
    const std::vector<std::string>& prng_names,
    std::uint64_t seed) {

    const int n_draws = static_cast<int>(observed.size());
    const auto obs_counts = draw_counts(observed);
    const auto obs_probs  = se::core::stats::empirical_frequency(observed);
    const auto unif       = uniform_probs();

    std::vector<PRNGFingerprintResult> out;
    out.reserve(prng_names.size());

    for (std::size_t idx = 0; idx < prng_names.size(); ++idx) {
        const auto& name = prng_names[idx];

        se::core::MCConfig mc;
        mc.seed = seed ^ (static_cast<std::uint64_t>(idx + 1) * 0x9E3779B97F4A7C15ULL);
        const auto sim_freq = se::core::simulate_uniform_frequency(n_draws, mc);

        std::vector<int> sim_counts(se::core::N_MAX);
        for (int i = 0; i < se::core::N_MAX; ++i)
            sim_counts[i] = static_cast<int>(std::round(sim_freq(i) * n_draws * se::core::N_MAIN));

        PRNGFingerprintResult r;
        r.prng_name = name;
        r.chi_square = se::core::stats::chi_square_uniform(sim_counts, se::core::N_MAX);

        std::vector<double> sim_probs(se::core::N_MAX);
        for (int i = 0; i < se::core::N_MAX; ++i) sim_probs[i] = sim_freq(i);
        r.ks_test = se::core::stats::ks_test_uniform(sim_probs);

        std::vector<int> bin_seq;
        bin_seq.reserve(n_draws);
        for (const auto& d : observed) bin_seq.push_back(d.main[0] <= 45 ? 0 : 1);
        r.runs_test = se::core::stats::runs_test(bin_seq);

        std::vector<double> obs_p(obs_probs.data(), obs_probs.data() + obs_probs.size());
        r.js_divergence_vs_observed = se::core::stats::js_divergence(obs_p, sim_probs);

        r.overall_score = overall_score(r);
        out.push_back(std::move(r));
    }
    std::sort(out.begin(), out.end(),
              [](const PRNGFingerprintResult& a, const PRNGFingerprintResult& b) {
                  return a.overall_score > b.overall_score;
              });
    return out;
}

}  // namespace se::models
