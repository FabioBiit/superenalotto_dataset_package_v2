#include "generation/mmr_selector.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>

namespace se::generation {

double jaccard_distance(const Combination& a, const Combination& b) {
    std::set<int> sa(a.main.begin(), a.main.end());
    std::set<int> sb(b.main.begin(), b.main.end());
    int inter = 0;
    for (int x : sa) if (sb.count(x)) ++inter;
    const int uni = static_cast<int>(sa.size() + sb.size()) - inter;
    if (uni == 0) return 0.0;
    return 1.0 - static_cast<double>(inter) / static_cast<double>(uni);
}

std::vector<Combination> select_mmr(const std::vector<Combination>& candidates,
                                     const MMRConfig& cfg) {
    if (candidates.empty()) return {};
    if (cfg.k_final <= 0) return {};
    if (cfg.lambda < 0.0 || cfg.lambda > 1.0)
        throw std::invalid_argument("lambda must be in [0, 1]");

    const int k = std::min(cfg.k_final, static_cast<int>(candidates.size()));

    std::vector<Combination> pool = candidates;
    std::sort(pool.begin(), pool.end(),
              [](const Combination& a, const Combination& b) { return a.score > b.score; });

    std::vector<Combination> selected;
    selected.reserve(k);
    selected.push_back(pool.front());
    pool.erase(pool.begin());

    while (static_cast<int>(selected.size()) < k && !pool.empty()) {
        double best_mmr = -1e9;
        int best_idx = -1;
        const double score_max = std::max_element(pool.begin(), pool.end(),
            [](const Combination& a, const Combination& b) { return a.score < b.score; })->score;
        const double score_min = std::min_element(pool.begin(), pool.end(),
            [](const Combination& a, const Combination& b) { return a.score < b.score; })->score;
        const double score_range = std::max(score_max - score_min, 1e-9);

        for (std::size_t i = 0; i < pool.size(); ++i) {
            const double norm_score = (pool[i].score - score_min) / score_range;
            double min_dist = 1.0;
            for (const auto& s : selected)
                min_dist = std::min(min_dist, jaccard_distance(pool[i], s));
            const double mmr = cfg.lambda * norm_score + (1.0 - cfg.lambda) * min_dist;
            if (mmr > best_mmr) {
                best_mmr = mmr;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx < 0) break;
        selected.push_back(pool[best_idx]);
        pool.erase(pool.begin() + best_idx);
    }

    return selected;
}

}  // namespace se::generation
