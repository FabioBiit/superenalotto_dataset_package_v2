#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "backtesting/walk_forward.hpp"
#include "core/types.hpp"

#include <chrono>

using namespace se;
using Catch::Matchers::WithinAbs;

namespace {

core::DrawSet make_synthetic_draws(int N, std::uint64_t seed = 11) {
    core::DrawSet d;
    d.reserve(N);
    std::uint64_t s = seed;
    auto next = [&]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s;
    };
    for (int i = 0; i < N; ++i) {
        core::Draw row;
        row.contest_number = i + 1;
        row.date = std::chrono::year_month_day{
            std::chrono::year{2009 + i / 156},
            std::chrono::month{static_cast<unsigned>(1 + (i % 12))},
            std::chrono::day{static_cast<unsigned>(1 + (i % 28))}
        };
        // sample 6 distinct in 1..90
        std::array<bool, core::N_MAX> used{};
        for (int k = 0; k < core::N_MAIN; ++k) {
            int n;
            do { n = 1 + static_cast<int>(next() % core::N_MAX); } while (used[n - 1]);
            used[n - 1] = true;
            row.main[k] = n;
        }
        std::sort(row.main.begin(), row.main.end());
        row.jolly     = 1 + static_cast<int>(next() % core::N_MAX);
        row.superstar = 1 + static_cast<int>(next() % core::N_MAX);
        d.push_back(row);
    }
    return d;
}

}  // namespace

TEST_CASE("build_indicator: shape and row-sum", "[backtesting]") {
    auto draws = make_synthetic_draws(50);
    auto Y = backtesting::WalkForward::build_indicator(draws);
    REQUIRE(Y.rows() == 50);
    REQUIRE(Y.cols() == core::N_MAX);
    for (int t = 0; t < Y.rows(); ++t) REQUIRE(Y.row(t).sum() == core::N_MAIN);
}

TEST_CASE("qmat_G0: uniform 6/90", "[backtesting]") {
    auto draws = make_synthetic_draws(100);
    backtesting::WalkForward wf{{50, 50}};
    auto q = wf.qmat_G0(draws);
    REQUIRE(q.rows() == 100);
    REQUIRE(q.cols() == core::N_MAX);
    REQUIRE_THAT(q(0, 0), WithinAbs(6.0 / 90.0, 1e-12));
    REQUIRE_THAT(q.row(99).sum(), WithinAbs(6.0, 1e-9));
}

TEST_CASE("qmat_G2: rows approximately sum to 6", "[backtesting]") {
    auto draws = make_synthetic_draws(200);
    backtesting::WalkForward wf{{50, 50}};
    auto q = wf.qmat_G2(draws);
    for (int t = 50; t < q.rows(); ++t)
        REQUIRE_THAT(q.row(t).sum(), WithinAbs(6.0, 1e-6));
}

TEST_CASE("evaluate: hits in [0, 6]", "[backtesting]") {
    auto draws = make_synthetic_draws(150);
    backtesting::WalkForward wf{{50, 50}};
    auto q = wf.qmat_G0(draws);
    auto r = wf.evaluate("G0", q, draws, 0);
    REQUIRE(r.n_test == 100);
    REQUIRE(r.per_draw_hits.minCoeff() >= 0);
    REQUIRE(r.per_draw_hits.maxCoeff() <= 6);
    REQUIRE(r.hits_distribution.sum() == 100);
}

TEST_CASE("paired_permutation_test: identical -> p=1", "[backtesting]") {
    Eigen::VectorXi a(50); a.setConstant(2);
    Eigen::VectorXi b(50); b.setConstant(2);
    backtesting::WalkForward wf;
    auto pr = wf.paired_permutation_test(a, b, 1000);
    REQUIRE_THAT(pr.delta_hits_vs_G0, WithinAbs(0.0, 1e-12));
    REQUIRE(pr.p_value >= 0.0);
    REQUIRE_FALSE(pr.robustly_beats_G0);
}

TEST_CASE("paired_permutation_test: a > b consistently -> p small", "[backtesting]") {
    Eigen::VectorXi a(200); a.setConstant(3);
    Eigen::VectorXi b(200); b.setConstant(1);
    backtesting::WalkForward wf;
    auto pr = wf.paired_permutation_test(a, b, 2000);
    REQUIRE(pr.delta_hits_vs_G0 > 0);
    REQUIRE(pr.p_value < 0.01);
    REQUIRE(pr.robustly_beats_G0);
}
