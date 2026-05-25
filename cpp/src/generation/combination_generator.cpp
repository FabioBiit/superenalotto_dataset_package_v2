#include "generation/combination_generator.hpp"

#include "core/prng.hpp"
#include "core/stats.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>

namespace se::generation {

namespace {

using Mask = std::array<bool, se::core::N_MAX>;

bool has_pattern_issues(const std::array<int, se::core::N_MAIN>& sorted) {
    bool all_even = true, all_odd = true;
    for (int v : sorted) { if (v % 2 == 0) all_odd = false; else all_even = false; }
    if (all_even || all_odd) return true;

    std::set<int> decs;
    for (int v : sorted) decs.insert((v - 1) / 10);
    if (decs.size() <= 1) return true;

    int max_consec = 1, cur = 1;
    for (int i = 1; i < se::core::N_MAIN; ++i) {
        if (sorted[i] == sorted[i - 1] + 1) { ++cur; max_consec = std::max(max_consec, cur); }
        else cur = 1;
    }
    if (max_consec >= 4) return true;

    bool all_low = true, all_high = true;
    for (int v : sorted) {
        if (v > 30) all_low = false;
        if (v < 61) all_high = false;
    }
    if (all_low || all_high) return true;
    return false;
}

std::array<int, se::core::N_MAIN> sample_by_weights(se::core::PRNG& prng,
                                                      const Eigen::VectorXd& weights) {
    std::array<int, se::core::N_MAIN> out{};
    Eigen::VectorXd w = weights;
    for (int k = 0; k < se::core::N_MAIN; ++k) {
        const double total = w.sum();
        if (total <= 0) { out[k] = k + 1; continue; }
        const double r = prng.next_unit() * total;
        double cum = 0.0;
        int chosen = -1;
        for (int i = 0; i < w.size(); ++i) {
            cum += w(i);
            if (cum >= r) { chosen = i; break; }
        }
        if (chosen < 0) chosen = static_cast<int>(w.size()) - 1;
        out[k] = chosen + 1;
        w(chosen) = 0.0;
    }
    std::sort(out.begin(), out.end());
    return out;
}

double quality_score(const std::array<int, se::core::N_MAIN>& sorted,
                      const Eigen::VectorXd& freq,
                      const Eigen::VectorXd& delay,
                      const std::set<int>& recent10) {
    double score = 0.0;
    int n_even = 0;
    for (int v : sorted) if (v % 2 == 0) ++n_even;
    if (n_even == 3) score += 15.0;
    else if (n_even == 2 || n_even == 4) score += 10.0;

    std::set<int> decs;
    for (int v : sorted) decs.insert((v - 1) / 10);
    if (decs.size() >= 5) score += 15.0;
    else if (decs.size() == 4) score += 10.0;

    int n_low = 0, n_mid = 0, n_high = 0;
    for (int v : sorted) {
        if (v <= 30) ++n_low;
        else if (v <= 60) ++n_mid;
        else ++n_high;
    }
    if (n_low >= 1 && n_mid >= 1 && n_high >= 1) score += 12.0;

    int overlap_recent = 0;
    for (int v : sorted) if (recent10.count(v)) ++overlap_recent;
    score += (6 - overlap_recent) * 2.0;

    int max_consec = 1, cur = 1;
    for (int i = 1; i < se::core::N_MAIN; ++i) {
        if (sorted[i] == sorted[i - 1] + 1) { ++cur; max_consec = std::max(max_consec, cur); }
        else cur = 1;
    }
    score -= (max_consec - 1) * 3.0;

    if (!has_pattern_issues(sorted)) score += 10.0;

    const int range = sorted.back() - sorted.front();
    if (range >= 40 && range <= 75) score += 8.0;

    double freq_sum  = 0.0;
    double delay_sum = 0.0;
    for (int v : sorted) {
        freq_sum  += freq(v - 1);
        delay_sum += delay(v - 1);
    }
    const double freq_mean = freq_sum / se::core::N_MAIN;
    const double exp_freq  = 1.0 / se::core::N_MAX;
    score += 5.0 * (1.0 - std::abs(freq_mean - exp_freq) / exp_freq);
    if (delay_sum / se::core::N_MAIN >= 15.0 && delay_sum / se::core::N_MAIN <= 40.0) score += 5.0;

    return std::clamp(score, 0.0, 100.0);
}

std::array<int, se::core::N_MAIN> sample_anti_recent(se::core::PRNG& prng,
                                                       const std::set<int>& recent10) {
    Eigen::VectorXd w = Eigen::VectorXd::Ones(se::core::N_MAX);
    for (int v : recent10) if (v >= 1 && v <= se::core::N_MAX) w(v - 1) = 0.05;
    return sample_by_weights(prng, w);
}

std::array<int, se::core::N_MAIN> sample_anti_pattern_balanced(se::core::PRNG& prng) {
    auto pick_in = [&](int lo, int hi, std::set<int>& used, int count) {
        std::vector<int> avail;
        for (int v = lo; v <= hi; ++v) if (!used.count(v)) avail.push_back(v);
        for (int k = 0; k < count && !avail.empty(); ++k) {
            const int j = static_cast<int>(prng.next_u64() % avail.size());
            used.insert(avail[j]);
            avail.erase(avail.begin() + j);
        }
    };
    std::set<int> picked;
    pick_in(1,  30, picked, 2);
    pick_in(31, 60, picked, 2);
    pick_in(61, 90, picked, 2);
    while (picked.size() < se::core::N_MAIN) {
        const int v = 1 + static_cast<int>(prng.next_u64() % se::core::N_MAX);
        picked.insert(v);
    }
    std::array<int, se::core::N_MAIN> out{};
    int k = 0;
    for (int v : picked) { if (k < se::core::N_MAIN) out[k++] = v; }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

std::string strategy_name(Strategy s) {
    switch (s) {
        case Strategy::AntiPatternBalanced: return "anti_pattern_balanced";
        case Strategy::AntiRecent:          return "anti_recent";
        case Strategy::DelayWeighted:       return "delay_weighted";
        case Strategy::FreqWeighted:        return "freq_weighted";
        case Strategy::MixedBalance:        return "mixed_balance";
        case Strategy::MonteCarloUniform:   return "monte_carlo_uniform";
    }
    return "unknown";
}

std::vector<Combination> generate_candidates(
    const se::core::DrawSet& history,
    const Eigen::VectorXd& number_weights,
    const GeneratorConfig& cfg) {

    if (history.empty()) throw std::invalid_argument("history is empty");

    auto prng = se::core::make_prng("PCG64", cfg.seed);

    const auto freq  = se::core::stats::empirical_frequency(history);
    const auto delay = se::core::stats::delays(history);

    std::set<int> recent10;
    const int start = std::max(0, static_cast<int>(history.size()) - cfg.overlap_window);
    for (int i = start; i < static_cast<int>(history.size()); ++i)
        for (int v : history[i].main) recent10.insert(v);

    Eigen::VectorXd uniform_w = Eigen::VectorXd::Ones(se::core::N_MAX);
    Eigen::VectorXd freq_w    = freq.array() + 1e-6;
    Eigen::VectorXd delay_w   = delay.array() + 1.0;
    Eigen::VectorXd mixed_w   = freq_w.array() * (delay_w.array() / delay_w.maxCoeff()).sqrt();

    if (number_weights.size() == se::core::N_MAX) {
        freq_w  = freq_w.array()  * number_weights.array();
        delay_w = delay_w.array() * number_weights.array();
        mixed_w = mixed_w.array() * number_weights.array();
    }

    std::vector<std::pair<Strategy, Eigen::VectorXd>> base = {
        {Strategy::MonteCarloUniform, uniform_w},
        {Strategy::FreqWeighted,      freq_w},
        {Strategy::DelayWeighted,     delay_w},
        {Strategy::MixedBalance,      mixed_w},
    };

    std::vector<Combination> all;
    all.reserve(cfg.n_per_strategy * 6);

    std::set<std::array<int, se::core::N_MAIN>> seen;

    auto try_add = [&](const std::array<int, se::core::N_MAIN>& sorted, Strategy strat) {
        if (has_pattern_issues(sorted)) return;
        if (!seen.insert(sorted).second) return;
        Combination c;
        c.main     = sorted;
        c.strategy = strat;
        c.score    = quality_score(sorted, freq, delay, recent10);
        all.push_back(c);
    };

    for (const auto& [strat, w] : base) {
        int safety = 0;
        while (static_cast<int>(std::count_if(all.begin(), all.end(),
              [&](const Combination& c) { return c.strategy == strat; })) < cfg.n_per_strategy
              && safety++ < cfg.n_per_strategy * 50) {
            const auto candidate = sample_by_weights(*prng, w);
            try_add(candidate, strat);
        }
    }

    int safety = 0;
    while (static_cast<int>(std::count_if(all.begin(), all.end(),
          [](const Combination& c) { return c.strategy == Strategy::AntiRecent; })) < cfg.n_per_strategy
          && safety++ < cfg.n_per_strategy * 50) {
        try_add(sample_anti_recent(*prng, recent10), Strategy::AntiRecent);
    }

    safety = 0;
    while (static_cast<int>(std::count_if(all.begin(), all.end(),
          [](const Combination& c) { return c.strategy == Strategy::AntiPatternBalanced; })) < cfg.n_per_strategy
          && safety++ < cfg.n_per_strategy * 50) {
        try_add(sample_anti_pattern_balanced(*prng), Strategy::AntiPatternBalanced);
    }

    // Pick jolly/superstar from history top-15 of each, excluding main numbers.
    std::array<int, se::core::N_MAX + 1> jolly_freq{}, ss_freq{};
    for (const auto& d : history) {
        if (d.jolly     >= 1 && d.jolly     <= se::core::N_MAX) ++jolly_freq[d.jolly];
        if (d.superstar >= 1 && d.superstar <= se::core::N_MAX) ++ss_freq[d.superstar];
    }
    std::vector<int> jolly_top, ss_top;
    for (int i = 1; i <= se::core::N_MAX; ++i) { jolly_top.push_back(i); ss_top.push_back(i); }
    std::sort(jolly_top.begin(), jolly_top.end(),
              [&](int a, int b) { return jolly_freq[a] > jolly_freq[b]; });
    std::sort(ss_top.begin(), ss_top.end(),
              [&](int a, int b) { return ss_freq[a] > ss_freq[b]; });
    jolly_top.resize(std::min<std::size_t>(15, jolly_top.size()));
    ss_top.resize(std::min<std::size_t>(15, ss_top.size()));

    for (auto& c : all) {
        std::set<int> main_set(c.main.begin(), c.main.end());
        for (int j : jolly_top) if (!main_set.count(j)) { c.jolly = j; break; }
        for (int s : ss_top)    if (!main_set.count(s) && s != c.jolly) { c.superstar = s; break; }
        if (c.jolly     == 0) c.jolly     = (c.main[0] == 1 ? 2 : 1);
        if (c.superstar == 0) c.superstar = (c.main[0] == 3 ? 4 : 3);
    }

    return all;
}

}  // namespace se::generation
