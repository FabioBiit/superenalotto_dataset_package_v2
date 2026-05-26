#include "backtesting/walk_forward.hpp"

#include "core/prng.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef SE_HAS_OPENMP
#include <omp.h>
#endif

namespace se::backtesting {

namespace {

using se::core::N_MAIN;
using se::core::N_MAX;

constexpr auto CP_FLUTTER_Y = std::chrono::year{2022};
constexpr auto CP_FLUTTER_M = std::chrono::August;
constexpr auto CP_FLUTTER_D = std::chrono::day{4};

constexpr auto CP_4WEEK_Y = std::chrono::year{2023};
constexpr auto CP_4WEEK_M = std::chrono::July;
constexpr auto CP_4WEEK_D = std::chrono::day{7};

inline bool date_lt(const std::chrono::year_month_day& a,
                     const std::chrono::year_month_day& b) {
    return std::chrono::sys_days{a} < std::chrono::sys_days{b};
}

}  // namespace

WalkForward::WalkForward(WalkForwardConfig cfg) : cfg_(cfg) {}

Eigen::MatrixXi WalkForward::build_indicator(const se::core::DrawSet& draws) {
    const int N = static_cast<int>(draws.size());
    Eigen::MatrixXi Y = Eigen::MatrixXi::Zero(N, N_MAX);
    for (int t = 0; t < N; ++t) {
        for (int n : draws[t].main) Y(t, n - 1) = 1;
    }
    return Y;
}

Eigen::VectorXi WalkForward::build_regimes(const se::core::DrawSet& draws) {
    const int N = static_cast<int>(draws.size());
    Eigen::VectorXi r(N);
    const std::chrono::year_month_day flutter{CP_FLUTTER_Y, CP_FLUTTER_M, CP_FLUTTER_D};
    const std::chrono::year_month_day fourweek{CP_4WEEK_Y, CP_4WEEK_M, CP_4WEEK_D};
    for (int t = 0; t < N; ++t) {
        const auto& d = draws[t].date;
        r(t) = date_lt(d, flutter) ? 0 : (date_lt(d, fourweek) ? 1 : 2);
    }
    return r;
}

Eigen::MatrixXd WalkForward::qmat_G0(const se::core::DrawSet& draws) const {
    const int N = static_cast<int>(draws.size());
    return Eigen::MatrixXd::Constant(N, N_MAX, static_cast<double>(N_MAIN) / N_MAX);
}

Eigen::MatrixXd WalkForward::qmat_G2(const se::core::DrawSet& draws, double alpha) const {
    const Eigen::MatrixXi Y = build_indicator(draws);
    const int N = static_cast<int>(draws.size());
    Eigen::MatrixXd q(N, N_MAX);
    Eigen::RowVectorXd cum = Eigen::RowVectorXd::Zero(N_MAX);
    for (int t = 0; t < N; ++t) {
        const double total = cum.sum() + N_MAX * alpha;
        q.row(t) = (cum.array() + alpha) / total * N_MAIN;
        cum += Y.row(t).cast<double>();
    }
    return q;
}

Eigen::MatrixXd WalkForward::qmat_G3(const se::core::DrawSet& draws, int lag, double alpha) const {
    const Eigen::MatrixXi Y = build_indicator(draws);
    const int N = static_cast<int>(draws.size());
    Eigen::MatrixXd q = Eigen::MatrixXd::Constant(N, N_MAX, static_cast<double>(N_MAIN) / N_MAX);

    Eigen::ArrayXd n11(N_MAX), n10(N_MAX), n01(N_MAX), n00(N_MAX);
    n11.setZero(); n10.setZero(); n01.setZero(); n00.setZero();

    const int start = std::max(lag, cfg_.initial_train);

    // Pre-accumulate counts up to start-1 (inclusive).
    for (int t = lag; t < start; ++t) {
        for (int i = 0; i < N_MAX; ++i) {
            const int prev = Y(t - lag, i);
            const int curr = Y(t,        i);
            if (prev == 1 && curr == 1) n11(i) += 1.0;
            else if (prev == 1 && curr == 0) n10(i) += 1.0;
            else if (prev == 0 && curr == 1) n01(i) += 1.0;
            else                              n00(i) += 1.0;
        }
    }

    for (int t = start; t < N; ++t) {
        const Eigen::ArrayXd p_if_1 = (n11 + alpha) / (n11 + n10 + 2 * alpha);
        const Eigen::ArrayXd p_if_0 = (n01 + alpha) / (n01 + n00 + 2 * alpha);
        Eigen::ArrayXd p(N_MAX);
        for (int i = 0; i < N_MAX; ++i)
            p(i) = (Y(t - 1, i) == 1) ? p_if_1(i) : p_if_0(i);
        const double s = p.sum();
        if (s > 0) p = p / s * N_MAIN;
        q.row(t) = p.matrix().transpose();
        for (int i = 0; i < N_MAX; ++i) {
            const int prev = Y(t - lag, i);
            const int curr = Y(t,        i);
            if (prev == 1 && curr == 1) n11(i) += 1.0;
            else if (prev == 1 && curr == 0) n10(i) += 1.0;
            else if (prev == 0 && curr == 1) n01(i) += 1.0;
            else                              n00(i) += 1.0;
        }
    }
    return q;
}

Eigen::MatrixXd WalkForward::qmat_G4(const se::core::DrawSet& draws) const {
    const Eigen::MatrixXi Y = build_indicator(draws);
    const int N = static_cast<int>(draws.size());

    Eigen::MatrixXd X(N_MAX, 15);
    for (int n = 1; n <= N_MAX; ++n) {
        const int dec = (n - 1) / 10;
        const int par = n % 2;
        const int fas = (n <= 30) ? 0 : (n <= 60 ? 1 : 2);
        Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(15);
        if (dec >= 0 && dec < 9) row(dec) = 1.0;
        row(9 + par) = 1.0;
        row(11 + fas) = 1.0;
        row(14) = 1.0;
        X.row(n - 1) = row;
    }

    Eigen::MatrixXd q = Eigen::MatrixXd::Constant(N, N_MAX, static_cast<double>(N_MAIN) / N_MAX);
    int last_refit = -1;
    Eigen::VectorXd beta;
    Eigen::ArrayXd  counts = Eigen::ArrayXd::Zero(N_MAX);
    for (int t = 0; t < cfg_.initial_train; ++t)
        counts += Y.row(t).cast<double>().transpose().array();

    for (int t = cfg_.initial_train; t < N; ++t) {
        if (last_refit < 0 || (t - last_refit) >= cfg_.refit_every) {
            const Eigen::ArrayXd p_emp = (counts + 1.0) / (counts.sum() + N_MAX);
            const Eigen::ArrayXd logit_emp = (p_emp / (1.0 - p_emp)).log();
            // Rank-revealing solver (X has redundant columns: decade/parity/fascia
            // sum-to-one constraints + intercept = 3 redundancies).
            beta = X.completeOrthogonalDecomposition().solve(logit_emp.matrix());
            last_refit = t;
        }
        const Eigen::ArrayXd logit_pred = (X * beta).array();
        Eigen::ArrayXd p = 1.0 / (1.0 + (-logit_pred).exp());
        const double s = p.sum();
        if (s > 0) p = p / s * N_MAIN;
        p = p.cwiseMax(1e-6).cwiseMin(1.0 - 1e-6);
        q.row(t) = p.matrix().transpose();
        counts += Y.row(t).cast<double>().transpose().array();
    }
    return q;
}

Eigen::MatrixXd WalkForward::qmat_G5(const se::core::DrawSet& draws, double alpha) const {
    const Eigen::MatrixXi Y = build_indicator(draws);
    const Eigen::VectorXi regimes = build_regimes(draws);
    const int N = static_cast<int>(draws.size());
    Eigen::MatrixXd q = Eigen::MatrixXd::Constant(N, N_MAX, static_cast<double>(N_MAIN) / N_MAX);
    for (int t = 1; t < N; ++t) {
        const int reg = regimes(t);
        Eigen::ArrayXd counts = Eigen::ArrayXd::Zero(N_MAX);
        int same_count = 0;
        for (int u = 0; u < t; ++u) {
            if (regimes(u) == reg) {
                counts += Y.row(u).cast<double>().transpose().array();
                ++same_count;
            }
        }
        if (same_count < 30) {
            counts.setZero();
            for (int u = 0; u < t; ++u) counts += Y.row(u).cast<double>().transpose().array();
        }
        const double total = counts.sum() + N_MAX * alpha;
        q.row(t) = ((counts + alpha) / total * N_MAIN).matrix().transpose();
    }
    return q;
}

Eigen::MatrixXd WalkForward::qmat_G6(const Eigen::MatrixXd& g2,
                                       const Eigen::MatrixXd& g3,
                                       const Eigen::MatrixXd& g4,
                                       std::array<double, 3> w) const {
    return w[0] * g2 + w[1] * g3 + w[2] * g4;
}

ModelResult WalkForward::evaluate(const std::string& name,
                                    const Eigen::MatrixXd& q,
                                    const se::core::DrawSet& draws,
                                    int k_params) const {
    const Eigen::MatrixXi Y = build_indicator(draws);
    const int N = static_cast<int>(draws.size());
    const int t0 = cfg_.initial_train;
    const int n_test = N - t0;
    if (n_test <= 0) throw std::runtime_error("WalkForward::evaluate: train >= N");

    const double eps = 1e-9;
    Eigen::MatrixXd qe = q.bottomRows(n_test);
    Eigen::MatrixXi ye = Y.bottomRows(n_test);
    qe = qe.cwiseMax(eps).cwiseMin(1.0 - eps);

    double ll    = 0.0;
    double brier = 0.0;
    Eigen::VectorXi per_draw_hits(n_test);
    Eigen::VectorXi hits_dist = Eigen::VectorXi::Zero(7);

    for (int t = 0; t < n_test; ++t) {
        std::vector<int> top6_idx(N_MAX);
        for (int i = 0; i < N_MAX; ++i) top6_idx[i] = i;
        std::partial_sort(top6_idx.begin(), top6_idx.begin() + N_MAIN, top6_idx.end(),
            [&](int a, int b) { return qe(t, a) > qe(t, b); });

        int hits = 0;
        for (int k = 0; k < N_MAIN; ++k)
            if (ye(t, top6_idx[k]) == 1) ++hits;
        per_draw_hits(t) = hits;
        if (hits >= 0 && hits <= 6) ++hits_dist(hits);

        for (int i = 0; i < N_MAX; ++i) {
            const double y = ye(t, i);
            const double p = qe(t, i);
            ll    += y * std::log(p) + (1 - y) * std::log(1 - p);
            brier += (p - y) * (p - y);
        }
    }

    const double n_cells = static_cast<double>(n_test) * N_MAX;
    brier /= n_cells;

    ModelResult r;
    r.name              = name;
    r.k_params          = k_params;
    r.n_test            = n_test;
    r.log_likelihood    = ll;
    r.aic               = 2.0 * k_params - 2.0 * ll;
    r.bic               = std::log(n_cells) * k_params - 2.0 * ll;
    r.brier_score       = brier;
    r.avg_hits_at_6     = per_draw_hits.cast<double>().mean();
    const double m_     = r.avg_hits_at_6;
    double var          = 0.0;
    for (int t = 0; t < n_test; ++t) var += (per_draw_hits(t) - m_) * (per_draw_hits(t) - m_);
    r.hits_std          = std::sqrt(var / n_test);
    r.hits_distribution = hits_dist;
    r.per_draw_hits     = per_draw_hits;
    return r;
}

PermResult WalkForward::paired_permutation_test(const Eigen::VectorXi& hits_a,
                                                  const Eigen::VectorXi& hits_b,
                                                  int n_iter,
                                                  std::uint64_t seed) const {
    if (hits_a.size() != hits_b.size())
        throw std::invalid_argument("paired_permutation_test: size mismatch");
    const int n = static_cast<int>(hits_a.size());
    Eigen::VectorXd diff(n);
    for (int i = 0; i < n; ++i) diff(i) = static_cast<double>(hits_a(i) - hits_b(i));
    const double obs = diff.mean();
    auto prng = se::core::make_prng("PCG64", seed);

    int n_extreme = 0;
#ifdef SE_HAS_OPENMP
    #pragma omp parallel
    {
        int local_extreme = 0;
        auto local_prng = se::core::make_prng("PCG64", seed ^ static_cast<std::uint64_t>(omp_get_thread_num() + 1));
        #pragma omp for schedule(static)
        for (int k = 0; k < n_iter; ++k) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                const std::uint64_t r = local_prng->next_u64();
                const double sign = (r & 1ULL) ? 1.0 : -1.0;
                sum += sign * diff(i);
            }
            if (std::abs(sum / n) >= std::abs(obs)) ++local_extreme;
        }
        #pragma omp atomic
        n_extreme += local_extreme;
    }
#else
    for (int k = 0; k < n_iter; ++k) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            const std::uint64_t r = prng->next_u64();
            const double sign = (r & 1ULL) ? 1.0 : -1.0;
            sum += sign * diff(i);
        }
        if (std::abs(sum / n) >= std::abs(obs)) ++n_extreme;
    }
#endif

    PermResult r;
    r.delta_hits_vs_G0    = obs;
    r.p_value             = static_cast<double>(n_extreme) / n_iter;
    r.robustly_beats_G0   = (obs > 0.0) && (r.p_value < 0.05);
    return r;
}

WalkForward::FullReport WalkForward::run_all(const se::core::DrawSet& draws) const {
    FullReport rep;
    const auto qG0 = qmat_G0(draws);
    const auto qG2 = qmat_G2(draws);
    const auto qG3 = qmat_G3(draws);
    const auto qG4 = qmat_G4(draws);
    const auto qG5 = qmat_G5(draws);
    const auto qG6 = qmat_G6(qG2, qG3, qG4);

    rep.models.push_back(evaluate("G0_uniform",      qG0, draws,   0));
    rep.models.push_back(evaluate("G2_dirichlet",    qG2, draws,  90));
    rep.models.push_back(evaluate("G3_markov_lag1",  qG3, draws, 180));
    rep.models.push_back(evaluate("G4_structural",   qG4, draws,  14));
    rep.models.push_back(evaluate("G5_regime_split", qG5, draws, 270));
    rep.models.push_back(evaluate("G6_ensemble",     qG6, draws, 286));

    const double g0_hits = rep.models.front().avg_hits_at_6;
    for (auto& m : rep.models)
        m.lift_vs_uniform_pct = (g0_hits > 0 ? (m.avg_hits_at_6 / g0_hits - 1) * 100.0 : 0.0);

    const Eigen::VectorXi& hits_g0 = rep.models.front().per_draw_hits;
    const std::vector<std::string> targets = {
        "G2_dirichlet", "G3_markov_lag1", "G4_structural", "G5_regime_split", "G6_ensemble"
    };
    for (const auto& t : targets) {
        for (const auto& m : rep.models) {
            if (m.name == t) {
                auto pr = paired_permutation_test(m.per_draw_hits, hits_g0);
                rep.permutation_tests.emplace_back(t, pr);
                break;
            }
        }
    }

    const auto& best = *std::max_element(rep.models.begin(), rep.models.end(),
        [](const ModelResult& a, const ModelResult& b) { return a.avg_hits_at_6 < b.avg_hits_at_6; });
    bool any_robust = false;
    for (const auto& [_, pr] : rep.permutation_tests) if (pr.robustly_beats_G0) any_robust = true;
    rep.robust_signal = any_robust && (best.lift_vs_uniform_pct > 5.0);
    rep.verdict = rep.robust_signal
        ? ("Modello " + best.name + " mostra lift " + std::to_string(best.lift_vs_uniform_pct)
           + "% statisticamente rilevante. Verificare con out-of-sample esteso.")
        : "Dati compatibili con uniforme. Nessun modello batte G0 in modo robusto. "
          "Le combinazioni generate restano euristiche (non previsioni).";
    return rep;
}

}  // namespace se::backtesting
