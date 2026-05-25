#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "inference/model_selection.hpp"

using namespace se::inference;
using Catch::Matchers::WithinAbs;

TEST_CASE("AIC = 2k - 2logL", "[inference]") {
    auto m = compute_metrics("test", -100.0, 5, 1000);
    REQUIRE_THAT(m.aic, WithinAbs(2.0 * 5 - 2.0 * (-100.0), 1e-9));
}

TEST_CASE("BIC = ln(n)*k - 2logL", "[inference]") {
    auto m = compute_metrics("test", -100.0, 5, 1000);
    REQUIRE_THAT(m.bic, WithinAbs(std::log(1000.0) * 5 - 2.0 * (-100.0), 1e-9));
}

TEST_CASE("rank_models orders by BIC ascending", "[inference]") {
    auto m1 = compute_metrics("m1", -100.0, 5,  1000);   // BIC large
    auto m2 = compute_metrics("m2", -50.0,  3,  1000);   // BIC smaller
    auto m3 = compute_metrics("m3", -200.0, 10, 1000);
    auto ranked = rank_models({m1, m2, m3}, "BIC");
    REQUIRE(ranked.front().model_name == "m2");
}
