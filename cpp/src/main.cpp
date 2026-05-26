#include "core/data_loader.hpp"
#include "core/monte_carlo.hpp"
#include "core/stats.hpp"
#include "models/prng_fingerprint.hpp"
#include "generation/combination_generator.hpp"
#include "generation/mmr_selector.hpp"
#include "backtesting/walk_forward.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_version() {
    std::cout << "SuperEnalotto Engine v0.1.0\n"
              << "Build: " << __DATE__ << " " << __TIME__ << "\n"
              << "C++ std: " << __cplusplus << "\n";
#ifdef SE_HAS_OPENMP
    std::cout << "OpenMP: enabled\n";
#else
    std::cout << "OpenMP: disabled\n";
#endif
#ifdef SE_HAS_CUDA
    std::cout << "CUDA:   enabled\n";
#else
    std::cout << "CUDA:   disabled\n";
#endif
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  se_cli --version\n"
              << "  se_cli --validate    <csv_path>\n"
              << "  se_cli --frequency   <csv_path>\n"
              << "  se_cli --fingerprint <csv_path>\n"
              << "  se_cli --generate    <csv_path> [k_final=25]\n"
              << "  se_cli --backtest    <csv_path> [initial_train=800]\n";
}

int cmd_validate(const std::filesystem::path& path) {
    auto draws = se::core::DataLoader::load_csv(path);
    std::cout << "Loaded "  << draws.size() << " draws\n"
              << "Invalid " << se::core::DataLoader::count_invalid(draws) << "\n";
    return 0;
}

int cmd_frequency(const std::filesystem::path& path) {
    auto draws = se::core::DataLoader::load_csv(path);
    auto freq  = se::core::stats::empirical_frequency(draws);

    std::vector<std::pair<int, double>> sorted;
    sorted.reserve(se::core::N_MAX);
    for (int i = 0; i < freq.size(); ++i) sorted.emplace_back(i + 1, freq(i));
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "Loaded " << draws.size() << " draws\n";
    std::cout << "Top-10 by frequency:\n";
    for (int i = 0; i < 10; ++i)
        std::cout << "  " << std::setw(2) << sorted[i].first << "  "
                  << std::fixed << std::setprecision(5) << sorted[i].second << "\n";

    std::vector<int> counts(se::core::N_MAX);
    int total = 0;
    for (const auto& d : draws) for (int n : d.main) { ++counts[n - 1]; ++total; }
    auto r = se::core::stats::chi_square_uniform(counts, se::core::N_MAX);
    std::cout << "\nChi-square uniform: chi2=" << r.statistic
              << " df=" << r.df << " p=" << r.p_value << "\n";
    return 0;
}

int cmd_fingerprint(const std::filesystem::path& path) {
    auto draws = se::core::DataLoader::load_csv(path);
    std::cout << "Loaded " << draws.size() << " draws\n"
              << "Comparing observed against PCG64, Philox, ChaCha20...\n\n";

    auto results = se::models::PRNGFingerprint::compare_against_observed(
        draws, {"PCG64", "Philox", "ChaCha20"}, 42);

    std::cout << std::left << std::setw(12) << "PRNG"
              << std::setw(12) << "chi2_p"
              << std::setw(12) << "ks_p"
              << std::setw(12) << "runs_p"
              << std::setw(12) << "JS_div"
              << "score\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(12) << r.prng_name
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.chi_square.p_value
                  << std::setw(12) << r.ks_test.p_value
                  << std::setw(12) << r.runs_test.p_value
                  << std::setw(12) << r.js_divergence_vs_observed
                  << r.overall_score << "\n";
    }
    return 0;
}

int cmd_generate(const std::filesystem::path& path, int k_final) {
    auto draws = se::core::DataLoader::load_csv(path);
    std::cout << "Loaded " << draws.size() << " draws\n";

    se::generation::GeneratorConfig gen_cfg;
    gen_cfg.n_per_strategy = 8;
    gen_cfg.seed = 42;
    gen_cfg.overlap_window = 10;

    auto candidates = se::generation::generate_candidates(draws, {}, gen_cfg);
    std::cout << "Generated " << candidates.size() << " candidates\n";

    se::generation::MMRConfig mmr;
    mmr.k_final = k_final;
    mmr.lambda  = 0.65;
    auto final = se::generation::select_mmr(candidates, mmr);

    std::cout << "\nWARNING (CLAUDE.md): Le combinazioni generate non sono previsioni. "
              << "Nel SuperEnalotto ogni combinazione ha esattamente la stessa "
              << "probabilita teorica di 1/622.614.630.\n\n";

    std::cout << std::left << std::setw(5) << "Rank"
              << std::setw(24) << "Strategy"
              << std::setw(28) << "Combination"
              << std::setw(8) << "Jolly"
              << std::setw(8) << "SS"
              << "Score\n";
    int rank = 1;
    for (const auto& c : final) {
        std::cout << std::left << std::setw(5) << rank++
                  << std::setw(24) << se::generation::strategy_name(c.strategy);
        std::string combo;
        for (int i = 0; i < se::core::N_MAIN; ++i) {
            if (i) combo += '-';
            if (c.main[i] < 10) combo += '0';
            combo += std::to_string(c.main[i]);
        }
        std::cout << std::setw(28) << combo
                  << std::setw(8) << c.jolly
                  << std::setw(8) << c.superstar
                  << std::fixed << std::setprecision(1) << c.score << "\n";
    }
    return 0;
}

int cmd_backtest(const std::filesystem::path& path, int initial_train) {
    auto draws = se::core::DataLoader::load_csv(path);
    std::cout << "Loaded " << draws.size() << " draws\n";
    if (initial_train >= static_cast<int>(draws.size()))
        throw std::runtime_error("initial_train must be < n_draws");

    se::backtesting::WalkForwardConfig cfg;
    cfg.initial_train = initial_train;
    cfg.refit_every   = 100;
    se::backtesting::WalkForward wf(cfg);

    std::cout << "Running walk-forward G0..G6 (initial_train=" << initial_train << ")...\n";
    auto rep = wf.run_all(draws);

    std::cout << "\n"
              << std::left << std::setw(22) << "Model"
              << std::setw(8)  << "k"
              << std::setw(14) << "log_lik"
              << std::setw(14) << "AIC"
              << std::setw(14) << "BIC"
              << std::setw(12) << "avg_hits"
              << std::setw(10) << "lift_%"
              << "\n";
    for (const auto& m : rep.models) {
        std::cout << std::left << std::setw(22) << m.name
                  << std::setw(8)  << m.k_params
                  << std::setw(14) << std::fixed << std::setprecision(2) << m.log_likelihood
                  << std::setw(14) << m.aic
                  << std::setw(14) << m.bic
                  << std::setw(12) << std::setprecision(4) << m.avg_hits_at_6
                  << std::setw(10) << std::setprecision(2) << m.lift_vs_uniform_pct
                  << "\n";
    }
    std::cout << "\nPermutation test vs G0:\n";
    for (const auto& [name, pr] : rep.permutation_tests) {
        std::cout << "  " << std::left << std::setw(22) << name
                  << "delta=" << std::setw(8) << std::setprecision(4) << pr.delta_hits_vs_G0
                  << "p=" << std::setw(8) << std::setprecision(4) << pr.p_value
                  << (pr.robustly_beats_G0 ? " ROBUST" : "")
                  << "\n";
    }
    std::cout << "\nVerdict: " << rep.verdict << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { print_usage(); return EXIT_SUCCESS; }
        const std::string cmd = argv[1];
        if (cmd == "--version") { print_version(); return EXIT_SUCCESS; }
        if (cmd == "--validate"    && argc >= 3) return cmd_validate(argv[2]);
        if (cmd == "--frequency"   && argc >= 3) return cmd_frequency(argv[2]);
        if (cmd == "--fingerprint" && argc >= 3) return cmd_fingerprint(argv[2]);
        if (cmd == "--generate"    && argc >= 3) {
            const int k = (argc >= 4) ? std::atoi(argv[3]) : 25;
            return cmd_generate(argv[2], k);
        }
        if (cmd == "--backtest"    && argc >= 3) {
            const int it = (argc >= 4) ? std::atoi(argv[3]) : 800;
            return cmd_backtest(argv[2], it);
        }
        print_usage();
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
