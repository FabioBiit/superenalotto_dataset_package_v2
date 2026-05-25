#include "core/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace se::core::stats {

namespace {

constexpr double EPS = 1e-12;

double chi2_sf(double x, int df) {
    if (x <= 0.0 || df <= 0) return 1.0;
    const double a = df * 0.5;
    const double b = x * 0.5;
    const double gln = std::lgamma(a);
    if (b < a + 1.0) {
        double ap  = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int n = 1; n < 200; ++n) {
            ap  += 1.0;
            del *= b / ap;
            sum += del;
            if (std::abs(del) < std::abs(sum) * 1e-12) break;
        }
        const double gamser = sum * std::exp(-b + a * std::log(b) - gln);
        return std::clamp(1.0 - gamser, 0.0, 1.0);
    }
    double bb = b + 1.0 - a;
    double c  = 1.0 / 1e-300;
    double d  = 1.0 / bb;
    double h  = d;
    for (int i = 1; i < 200; ++i) {
        const double an = -static_cast<double>(i) * (i - a);
        bb += 2.0;
        d  = an * d + bb;
        if (std::abs(d) < 1e-300) d = 1e-300;
        c  = bb + an / c;
        if (std::abs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-12) break;
    }
    return std::clamp(std::exp(-b + a * std::log(b) - gln) * h, 0.0, 1.0);
}

double normal_sf(double z) noexcept {
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

}  // namespace

TestResult chi_square_uniform(std::span<const int> counts, int expected_buckets) {
    if (expected_buckets <= 0) throw std::invalid_argument("expected_buckets must be > 0");
    const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    const double exp_per = total / static_cast<double>(expected_buckets);
    double chi2 = 0.0;
    for (int c : counts) {
        const double diff = c - exp_per;
        chi2 += (diff * diff) / std::max(exp_per, EPS);
    }
    TestResult r;
    r.statistic = chi2;
    r.df        = expected_buckets - 1;
    r.p_value   = chi2_sf(chi2, r.df);
    r.reject_h0 = r.p_value < 0.05;
    return r;
}

TestResult ks_test_uniform(std::span<const double> samples) {
    const int n = static_cast<int>(samples.size());
    if (n < 2) return {0.0, 1.0, 0, false};

    std::vector<double> s(samples.begin(), samples.end());
    std::sort(s.begin(), s.end());

    double d = 0.0;
    for (int i = 0; i < n; ++i) {
        const double f_emp_up  = static_cast<double>(i + 1) / n;
        const double f_emp_lo  = static_cast<double>(i)     / n;
        const double f_theo    = std::clamp(s[i], 0.0, 1.0);
        d = std::max(d, std::max(std::abs(f_emp_up - f_theo), std::abs(f_theo - f_emp_lo)));
    }

    const double k_stat = (std::sqrt(static_cast<double>(n)) + 0.12 + 0.11 / std::sqrt(static_cast<double>(n))) * d;
    double p = 0.0;
    for (int i = 1; i < 100; ++i) {
        const double sign = (i % 2 == 1) ? 1.0 : -1.0;
        const double term = std::exp(-2.0 * i * i * k_stat * k_stat);
        p += sign * term;
        if (term < 1e-15) break;
    }
    p = std::clamp(2.0 * p, 0.0, 1.0);
    return {d, p, n, p < 0.05};
}

TestResult runs_test(std::span<const int> binary_sequence) {
    const int n = static_cast<int>(binary_sequence.size());
    if (n < 2) return {0.0, 1.0, 0, false};

    int n1 = 0, n0 = 0;
    for (int x : binary_sequence) (x == 1 ? n1 : n0)++;
    if (n1 == 0 || n0 == 0) return {0.0, 1.0, n, false};

    int runs = 1;
    for (int i = 1; i < n; ++i)
        if (binary_sequence[i] != binary_sequence[i - 1]) ++runs;

    const double mu     = 2.0 * n1 * n0 / static_cast<double>(n) + 1.0;
    const double sigma2 = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n) /
                          (static_cast<double>(n) * n * (n - 1));
    const double sigma  = std::sqrt(std::max(sigma2, EPS));
    const double z      = (runs - mu) / sigma;
    const double p      = 2.0 * normal_sf(std::abs(z));
    return {z, std::clamp(p, 0.0, 1.0), n, p < 0.05};
}

TestResult ljung_box(std::span<const double> series, int max_lag) {
    const int n = static_cast<int>(series.size());
    if (n < max_lag + 2 || max_lag <= 0) return {0.0, 1.0, 0, false};

    double mean = std::accumulate(series.begin(), series.end(), 0.0) / n;
    double var  = 0.0;
    for (double x : series) var += (x - mean) * (x - mean);
    var /= n;
    if (var < EPS) return {0.0, 1.0, max_lag, false};

    double Q = 0.0;
    for (int k = 1; k <= max_lag; ++k) {
        double num = 0.0;
        for (int i = k; i < n; ++i) num += (series[i] - mean) * (series[i - k] - mean);
        num /= n;
        const double rho = num / var;
        Q += (rho * rho) / (n - k);
    }
    Q *= static_cast<double>(n) * (n + 2);
    const double p = chi2_sf(Q, max_lag);
    return {Q, p, max_lag, p < 0.05};
}

double shannon_entropy(std::span<const double> probs) {
    double h = 0.0;
    for (double p : probs) if (p > EPS) h -= p * std::log2(p);
    return h;
}

double kl_divergence(std::span<const double> p, std::span<const double> q) {
    if (p.size() != q.size()) throw std::invalid_argument("p and q must have equal size");
    double kl = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i)
        if (p[i] > EPS) kl += p[i] * std::log(p[i] / std::max(q[i], EPS));
    return kl;
}

double js_divergence(std::span<const double> p, std::span<const double> q) {
    if (p.size() != q.size()) throw std::invalid_argument("p and q must have equal size");
    std::vector<double> m(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) m[i] = 0.5 * (p[i] + q[i]);
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m);
}

Eigen::VectorXd empirical_frequency(const DrawSet& draws) {
    Eigen::VectorXd f = Eigen::VectorXd::Zero(N_MAX);
    for (const auto& d : draws) for (int n : d.main) ++f(n - 1);
    if (f.sum() > 0) f /= f.sum();
    return f;
}

Eigen::VectorXd delays(const DrawSet& draws) {
    Eigen::VectorXd last_seen = Eigen::VectorXd::Constant(N_MAX, -1.0);
    int idx = 0;
    for (const auto& d : draws) {
        for (int n : d.main) last_seen(n - 1) = idx;
        ++idx;
    }
    Eigen::VectorXd delay(N_MAX);
    for (int i = 0; i < N_MAX; ++i)
        delay(i) = (last_seen(i) < 0) ? idx : (idx - 1 - last_seen(i));
    return delay;
}

Eigen::MatrixXd cooccurrence_matrix(const DrawSet& draws) {
    Eigen::MatrixXd co = Eigen::MatrixXd::Zero(N_MAX, N_MAX);
    for (const auto& d : draws) {
        for (int i = 0; i < N_MAIN; ++i) {
            for (int j = i + 1; j < N_MAIN; ++j) {
                const int a = d.main[i] - 1;
                const int b = d.main[j] - 1;
                co(a, b) += 1.0;
                co(b, a) += 1.0;
            }
        }
    }
    return co;
}

}  // namespace se::core::stats
