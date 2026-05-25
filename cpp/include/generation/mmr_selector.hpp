#pragma once

#include "generation/combination_generator.hpp"
#include <vector>

namespace se::generation {

struct MMRConfig {
    int    k_final = 25;
    double lambda  = 0.65;
};

[[nodiscard]] std::vector<Combination> select_mmr(const std::vector<Combination>& candidates,
                                                    const MMRConfig& cfg = {});

[[nodiscard]] double jaccard_distance(const Combination& a, const Combination& b);

}  // namespace se::generation
