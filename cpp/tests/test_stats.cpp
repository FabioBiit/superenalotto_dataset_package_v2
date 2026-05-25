#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "core/stats.hpp"

using namespace se::core;
using Catch::Matchers::WithinAbs;

TEST_CASE("Shannon entropy uniform", "[stats]") {
    std::vector<double> p(90, 1.0 / 90.0);
    double h = stats::shannon_entropy(p);
    REQUIRE_THAT(h, WithinAbs(std::log2(90.0), 1e-9));
}

TEST_CASE("Chi-square uniform expected counts", "[stats]") {
    std::vector<int> counts(90, 100);
    auto r = stats::chi_square_uniform(counts, 90);
    REQUIRE_THAT(r.statistic, WithinAbs(0.0, 1e-9));
    REQUIRE(r.df == 89);
}

TEST_CASE("KL divergence self is zero", "[stats]") {
    std::vector<double> p(90, 1.0 / 90.0);
    REQUIRE_THAT(stats::kl_divergence(p, p), WithinAbs(0.0, 1e-9));
}

TEST_CASE("JS divergence symmetric", "[stats]") {
    std::vector<double> p = {0.5, 0.3, 0.2};
    std::vector<double> q = {0.1, 0.2, 0.7};
    REQUIRE_THAT(stats::js_divergence(p, q), WithinAbs(stats::js_divergence(q, p), 1e-9));
}
