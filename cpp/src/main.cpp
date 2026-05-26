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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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
              << "  se_cli --generate    <csv_path> [k_final] [options]\n"
              << "      options for --generate:\n"
              << "        --k-final N         portfolio size            (default 25)\n"
              << "        --n-per-strategy N  candidates per strategy   (default 8)\n"
              << "        --seed N            PRNG seed                 (default 42)\n"
              << "        --mmr-lambda V      score vs diversity 0..1   (default 0.65)\n"
              << "        --output-csv PATH   write CSV with warning    (default: stdout only)\n"
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

struct GenArgs {
    std::filesystem::path csv_path;
    int           k_final        = 25;
    int           n_per_strategy = 8;
    std::uint64_t seed           = 42;
    double        mmr_lambda     = 0.65;
    std::string   output_csv;
};

GenArgs parse_gen_args(int argc, char** argv) {
    // Layout: argv[0]=se_cli  argv[1]=--generate  argv[2]=<csv>  [argv[3]=k_final or --flag]  ...
    GenArgs g;
    if (argc < 3) throw std::runtime_error("--generate requires <csv_path>");
    g.csv_path = argv[2];

    int i = 3;
    // backwards-compat: bare positional integer after csv = k_final
    if (i < argc && argv[i][0] != '-') {
        g.k_final = std::atoi(argv[i]);
        ++i;
    }
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_next = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(flag + " requires a value");
            return argv[++i];
        };
        if      (a == "--k-final")        g.k_final        = std::atoi(need_next(a).c_str());
        else if (a == "--n-per-strategy") g.n_per_strategy = std::atoi(need_next(a).c_str());
        else if (a == "--seed")           g.seed           = std::strtoull(need_next(a).c_str(), nullptr, 10);
        else if (a == "--mmr-lambda")     g.mmr_lambda     = std::atof(need_next(a).c_str());
        else if (a == "--output-csv")     g.output_csv     = need_next(a);
        else throw std::runtime_error("unknown --generate option: " + a);
    }
    if (g.k_final        <= 0) throw std::runtime_error("k-final must be > 0");
    if (g.n_per_strategy <= 0) throw std::runtime_error("n-per-strategy must be > 0");
    if (g.mmr_lambda < 0.0 || g.mmr_lambda > 1.0)
        throw std::runtime_error("mmr-lambda must be in [0, 1]");
    return g;
}

int cmd_generate(const GenArgs& a) {
    auto draws = se::core::DataLoader::load_csv(a.csv_path);
    std::cout << "Loaded " << draws.size() << " draws\n";
    std::cout << "Params: k_final=" << a.k_final
              << " n_per_strategy=" << a.n_per_strategy
              << " seed=" << a.seed
              << " mmr_lambda=" << a.mmr_lambda << "\n";

    se::generation::GeneratorConfig gen_cfg;
    gen_cfg.n_per_strategy = a.n_per_strategy;
    gen_cfg.seed           = a.seed;
    gen_cfg.overlap_window = 10;

    auto candidates = se::generation::generate_candidates(draws, {}, gen_cfg);
    std::cout << "Generated " << candidates.size() << " candidates (pool target: "
              << (a.n_per_strategy * 6) << ")\n";

    se::generation::MMRConfig mmr;
    mmr.k_final = a.k_final;
    mmr.lambda  = a.mmr_lambda;
    auto final = se::generation::select_mmr(candidates, mmr);
    std::cout << "Portfolio: " << final.size() << " combinations\n";

    const std::string warning =
        "Le combinazioni generate non sono previsioni. Nel SuperEnalotto ogni "
        "combinazione ha esattamente la stessa probabilita teorica di "
        "1/622.614.630.";
    std::cout << "\nWARNING (CLAUDE.md): " << warning << "\n\n";

    auto fmt_combo = [](const auto& c) {
        std::string s;
        for (int i = 0; i < se::core::N_MAIN; ++i) {
            if (i) s += '-';
            if (c.main[i] < 10) s += '0';
            s += std::to_string(c.main[i]);
        }
        return s;
    };

    std::cout << std::left << std::setw(5)  << "Rank"
                          << std::setw(24) << "Strategy"
                          << std::setw(22) << "Combination"
                          << std::setw(6)  << "J"
                          << std::setw(6)  << "SS"
                          << "Score\n";
    int rank = 1;
    for (const auto& c : final) {
        std::cout << std::left << std::setw(5)  << rank++
                              << std::setw(24) << se::generation::strategy_name(c.strategy)
                              << std::setw(22) << fmt_combo(c)
                              << std::setw(6)  << c.jolly
                              << std::setw(6)  << c.superstar
                              << std::fixed << std::setprecision(1) << c.score << "\n";
    }

    if (!a.output_csv.empty()) {
        std::ofstream out(a.output_csv);
        if (!out) throw std::runtime_error("cannot open output CSV: " + a.output_csv);
        out << "# WARNING: " << warning << "\n";
        out << "rank,strategy,combinazione,jolly,superstar,score\n";
        rank = 1;
        for (const auto& c : final) {
            out << rank++ << ","
                << se::generation::strategy_name(c.strategy) << ","
                << fmt_combo(c) << ","
                << c.jolly << ","
                << c.superstar << ","
                << std::fixed << std::setprecision(2) << c.score << "\n";
        }
        std::cout << "\nCSV written to: " << a.output_csv << "\n";
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
            return cmd_generate(parse_gen_args(argc, argv));
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
