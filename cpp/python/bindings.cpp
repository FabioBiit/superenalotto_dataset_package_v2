#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "core/data_loader.hpp"
#include "core/monte_carlo.hpp"
#include "core/prng.hpp"
#include "core/stats.hpp"
#include "core/types.hpp"

#include "models/dirichlet_mcmc.hpp"
#include "models/hawkes.hpp"
#include "models/hmm_regime.hpp"
#include "models/prng_fingerprint.hpp"
#include "models/wallenius.hpp"

#include "inference/mcmc_engine.hpp"
#include "inference/model_selection.hpp"
#include "inference/posterior_predictive.hpp"

#include "generation/combination_generator.hpp"
#include "generation/mmr_selector.hpp"

#include "backtesting/walk_forward.hpp"

#include <string>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(se_engine, m) {
    m.doc() = "SuperEnalotto Engine - C++ acceleration for reverse-engineering "
              "probabilistic analysis of the Italian SuperEnalotto draws";
    m.attr("__version__") = "0.1.0";
    m.attr("C_6_90")      = se::core::C_6_90;

    // ---------- core::types ----------
    py::enum_<se::core::Operatore>(m, "Operatore")
        .value("SisalApax",    se::core::Operatore::SisalApax)
        .value("SisalFlutter", se::core::Operatore::SisalFlutter);

    py::class_<se::core::Draw>(m, "Draw")
        .def_readonly("contest_number",    &se::core::Draw::contest_number)
        .def_readonly("main",              &se::core::Draw::main)
        .def_readonly("jolly",             &se::core::Draw::jolly)
        .def_readonly("superstar",         &se::core::Draw::superstar)
        .def_readonly("winners_6",         &se::core::Draw::winners_6)
        .def_readonly("prize_6",           &se::core::Draw::prize_6)
        .def_readonly("operatore",         &se::core::Draw::operatore)
        .def_readonly("regime_4week",      &se::core::Draw::regime_4week)
        .def_readonly("regime_post_DM214", &se::core::Draw::regime_post_DM214);

    // ---------- core::data_loader ----------
    m.def("load_csv", [](const std::string& path) {
        return se::core::DataLoader::load_csv(path);
    }, py::arg("path"), "Load extractions from CSV with enrichment metadata");
    m.def("count_invalid", &se::core::DataLoader::count_invalid,
          "Count invalid draws in a DrawSet");

    // ---------- core::stats ----------
    py::class_<se::core::stats::TestResult>(m, "TestResult")
        .def_readonly("statistic", &se::core::stats::TestResult::statistic)
        .def_readonly("p_value",   &se::core::stats::TestResult::p_value)
        .def_readonly("df",        &se::core::stats::TestResult::df)
        .def_readonly("reject_h0", &se::core::stats::TestResult::reject_h0);

    m.def("chi_square_uniform",
        [](const std::vector<int>& counts, int expected_buckets) {
            return se::core::stats::chi_square_uniform(counts, expected_buckets);
        }, py::arg("counts"), py::arg("expected_buckets"));
    m.def("ks_test_uniform",
        [](const std::vector<double>& s) { return se::core::stats::ks_test_uniform(s); },
        py::arg("samples"));
    m.def("runs_test",
        [](const std::vector<int>& seq) { return se::core::stats::runs_test(seq); },
        py::arg("binary_sequence"));
    m.def("ljung_box",
        [](const std::vector<double>& s, int max_lag) {
            return se::core::stats::ljung_box(s, max_lag);
        }, py::arg("series"), py::arg("max_lag"));

    m.def("shannon_entropy",
        [](const std::vector<double>& p) { return se::core::stats::shannon_entropy(p); });
    m.def("kl_divergence",
        [](const std::vector<double>& p, const std::vector<double>& q) {
            return se::core::stats::kl_divergence(p, q);
        });
    m.def("js_divergence",
        [](const std::vector<double>& p, const std::vector<double>& q) {
            return se::core::stats::js_divergence(p, q);
        });

    m.def("empirical_frequency", &se::core::stats::empirical_frequency);
    m.def("delays",               &se::core::stats::delays);
    m.def("cooccurrence_matrix",  &se::core::stats::cooccurrence_matrix);

    // ---------- core::monte_carlo ----------
    py::class_<se::core::MCConfig>(m, "MCConfig")
        .def(py::init<>())
        .def_readwrite("n_simulations", &se::core::MCConfig::n_simulations)
        .def_readwrite("seed",          &se::core::MCConfig::seed)
        .def_readwrite("n_threads",     &se::core::MCConfig::n_threads)
        .def_readwrite("use_cuda",      &se::core::MCConfig::use_cuda);

    m.def("simulate_uniform_frequency",  &se::core::simulate_uniform_frequency,
          py::arg("n_draws"), py::arg("cfg") = se::core::MCConfig{});
    m.def("simulate_weighted_frequency",
        [](const std::vector<double>& w, int n_draws, const se::core::MCConfig& cfg) {
            return se::core::simulate_weighted_frequency(w, n_draws, cfg);
        }, py::arg("weights"), py::arg("n_draws"), py::arg("cfg") = se::core::MCConfig{});
    m.def("permutation_pvalues",
        [](const std::vector<int>& obs, const std::vector<double>& baseline,
           int n_perm, const se::core::MCConfig& cfg) {
            return se::core::permutation_pvalues(obs, baseline, n_perm, cfg);
        }, py::arg("observed_counts"), py::arg("baseline_probs"),
           py::arg("n_permutations"), py::arg("cfg") = se::core::MCConfig{});

    // ---------- models::wallenius ----------
    py::class_<se::models::Wallenius>(m, "Wallenius")
        .def(py::init<int, int>(), py::arg("n_balls") = 90, py::arg("n_pick") = 6)
        .def("log_pmf",               &se::models::Wallenius::log_pmf)
        .def("sample_one",            &se::models::Wallenius::sample_one,
             py::arg("omega"), py::arg("seed"))
        .def("total_log_likelihood",  &se::models::Wallenius::total_log_likelihood);

    // ---------- models::dirichlet_mcmc ----------
    py::class_<se::models::DirichletMCMCConfig>(m, "DirichletMCMCConfig")
        .def(py::init<>())
        .def_readwrite("n_iterations",   &se::models::DirichletMCMCConfig::n_iterations)
        .def_readwrite("n_burn_in",      &se::models::DirichletMCMCConfig::n_burn_in)
        .def_readwrite("n_thin",         &se::models::DirichletMCMCConfig::n_thin)
        .def_readwrite("prior_alpha",    &se::models::DirichletMCMCConfig::prior_alpha)
        .def_readwrite("seed",           &se::models::DirichletMCMCConfig::seed)
        .def_readwrite("adapt_proposal", &se::models::DirichletMCMCConfig::adapt_proposal);

    py::class_<se::models::PosteriorSample>(m, "PosteriorSample")
        .def_readonly("omega_chain",          &se::models::PosteriorSample::omega_chain)
        .def_readonly("log_likelihood_chain", &se::models::PosteriorSample::log_likelihood_chain)
        .def_readonly("acceptance_rate",      &se::models::PosteriorSample::acceptance_rate)
        .def_readonly("n_effective",          &se::models::PosteriorSample::n_effective);

    py::class_<se::models::DirichletMCMC>(m, "DirichletMCMC")
        .def(py::init<se::models::DirichletMCMCConfig>(),
             py::arg("cfg") = se::models::DirichletMCMCConfig{})
        .def("fit",                   &se::models::DirichletMCMC::fit)
        .def("posterior_mean_omega",  &se::models::DirichletMCMC::posterior_mean_omega);

    // ---------- models::hmm ----------
    py::class_<se::models::HMMConfig>(m, "HMMConfig")
        .def(py::init<>())
        .def_readwrite("n_states", &se::models::HMMConfig::n_states)
        .def_readwrite("max_iter", &se::models::HMMConfig::max_iter)
        .def_readwrite("tol",      &se::models::HMMConfig::tol)
        .def_readwrite("seed",     &se::models::HMMConfig::seed);

    py::class_<se::models::HMMFit>(m, "HMMFit")
        .def_readonly("pi0",            &se::models::HMMFit::pi0)
        .def_readonly("transition",     &se::models::HMMFit::transition)
        .def_readonly("emission",       &se::models::HMMFit::emission)
        .def_readonly("log_likelihood", &se::models::HMMFit::log_likelihood)
        .def_readonly("n_iterations",   &se::models::HMMFit::n_iterations)
        .def_readonly("converged",      &se::models::HMMFit::converged)
        .def_readonly("viterbi_path",   &se::models::HMMFit::viterbi_path);

    py::class_<se::models::HMMRegime>(m, "HMMRegime")
        .def(py::init<se::models::HMMConfig>(), py::arg("cfg") = se::models::HMMConfig{})
        .def("fit_baum_welch", &se::models::HMMRegime::fit_baum_welch)
        .def("viterbi",        &se::models::HMMRegime::viterbi);

    // ---------- models::hawkes ----------
    py::class_<se::models::HawkesParams>(m, "HawkesParams")
        .def(py::init<>())
        .def_readwrite("mu",    &se::models::HawkesParams::mu)
        .def_readwrite("alpha", &se::models::HawkesParams::alpha)
        .def_readwrite("beta",  &se::models::HawkesParams::beta);
    py::class_<se::models::HawkesFit>(m, "HawkesFit")
        .def_readonly("params",         &se::models::HawkesFit::params)
        .def_readonly("log_likelihood", &se::models::HawkesFit::log_likelihood)
        .def_readonly("converged",      &se::models::HawkesFit::converged);
    m.def("hawkes_fit",          &se::models::Hawkes::fit,
          py::arg("event_times"), py::arg("max_iter") = 500);
    m.def("hawkes_intensity_at", &se::models::Hawkes::intensity_at);

    // ---------- models::prng_fingerprint ----------
    py::class_<se::models::PRNGFingerprintResult>(m, "PRNGFingerprintResult")
        .def_readonly("prng_name",                  &se::models::PRNGFingerprintResult::prng_name)
        .def_readonly("chi_square",                 &se::models::PRNGFingerprintResult::chi_square)
        .def_readonly("ks_test",                    &se::models::PRNGFingerprintResult::ks_test)
        .def_readonly("runs_test",                  &se::models::PRNGFingerprintResult::runs_test)
        .def_readonly("js_divergence_vs_observed",  &se::models::PRNGFingerprintResult::js_divergence_vs_observed)
        .def_readonly("overall_score",              &se::models::PRNGFingerprintResult::overall_score);

    m.def("compare_prngs_against_observed",
          &se::models::PRNGFingerprint::compare_against_observed,
          py::arg("observed"),
          py::arg("prng_names") = std::vector<std::string>{"PCG64", "Philox", "ChaCha20"},
          py::arg("seed") = 42);

    // ---------- inference::mcmc_engine ----------
    py::class_<se::inference::MHConfig>(m, "MHConfig")
        .def(py::init<>())
        .def_readwrite("n_iterations", &se::inference::MHConfig::n_iterations)
        .def_readwrite("n_burn_in",    &se::inference::MHConfig::n_burn_in)
        .def_readwrite("step_size",    &se::inference::MHConfig::step_size)
        .def_readwrite("adapt",        &se::inference::MHConfig::adapt)
        .def_readwrite("seed",         &se::inference::MHConfig::seed);
    py::class_<se::inference::MHResult>(m, "MHResult")
        .def_readonly("chain",           &se::inference::MHResult::chain)
        .def_readonly("log_post_chain",  &se::inference::MHResult::log_post_chain)
        .def_readonly("acceptance_rate", &se::inference::MHResult::acceptance_rate);
    m.def("metropolis_hastings", &se::inference::metropolis_hastings,
          py::arg("log_post"), py::arg("init"), py::arg("cfg") = se::inference::MHConfig{});

    // ---------- inference::posterior_predictive ----------
    py::class_<se::inference::PPCResult>(m, "PPCResult")
        .def_readonly("observed_stats",     &se::inference::PPCResult::observed_stats)
        .def_readonly("simulated_stats",    &se::inference::PPCResult::simulated_stats)
        .def_readonly("p_values_two_sided", &se::inference::PPCResult::p_values_two_sided)
        .def_readonly("p_values_lower",     &se::inference::PPCResult::p_values_lower)
        .def_readonly("p_values_upper",     &se::inference::PPCResult::p_values_upper);
    m.def("posterior_predictive_check", &se::inference::posterior_predictive_check,
          py::arg("observed"), py::arg("omega_posterior_samples"),
          py::arg("n_simulations_per_sample") = 1);

    // ---------- inference::model_selection ----------
    py::class_<se::inference::ModelMetrics>(m, "ModelMetrics")
        .def_readonly("model_name",     &se::inference::ModelMetrics::model_name)
        .def_readonly("log_likelihood", &se::inference::ModelMetrics::log_likelihood)
        .def_readonly("n_parameters",   &se::inference::ModelMetrics::n_parameters)
        .def_readonly("n_observations", &se::inference::ModelMetrics::n_observations)
        .def_readonly("aic",            &se::inference::ModelMetrics::aic)
        .def_readonly("bic",            &se::inference::ModelMetrics::bic)
        .def_readonly("waic",           &se::inference::ModelMetrics::waic)
        .def_readonly("loo_cv",         &se::inference::ModelMetrics::loo_cv);
    m.def("compute_metrics", &se::inference::compute_metrics,
          py::arg("name"), py::arg("log_likelihood"), py::arg("n_parameters"),
          py::arg("n_observations"),
          py::arg("pointwise_log_likelihood") = Eigen::VectorXd{});
    m.def("rank_models", &se::inference::rank_models,
          py::arg("models"), py::arg("criterion") = "BIC");

    // ---------- generation ----------
    py::enum_<se::generation::Strategy>(m, "Strategy")
        .value("AntiPatternBalanced", se::generation::Strategy::AntiPatternBalanced)
        .value("AntiRecent",          se::generation::Strategy::AntiRecent)
        .value("DelayWeighted",       se::generation::Strategy::DelayWeighted)
        .value("FreqWeighted",        se::generation::Strategy::FreqWeighted)
        .value("MixedBalance",        se::generation::Strategy::MixedBalance)
        .value("MonteCarloUniform",   se::generation::Strategy::MonteCarloUniform);

    py::class_<se::generation::Combination>(m, "Combination")
        .def_readonly("main",      &se::generation::Combination::main)
        .def_readonly("jolly",     &se::generation::Combination::jolly)
        .def_readonly("superstar", &se::generation::Combination::superstar)
        .def_readonly("score",     &se::generation::Combination::score)
        .def_readonly("strategy",  &se::generation::Combination::strategy);

    py::class_<se::generation::GeneratorConfig>(m, "GeneratorConfig")
        .def(py::init<>())
        .def_readwrite("n_per_strategy", &se::generation::GeneratorConfig::n_per_strategy)
        .def_readwrite("seed",           &se::generation::GeneratorConfig::seed)
        .def_readwrite("overlap_window", &se::generation::GeneratorConfig::overlap_window);

    py::class_<se::generation::MMRConfig>(m, "MMRConfig")
        .def(py::init<>())
        .def_readwrite("k_final", &se::generation::MMRConfig::k_final)
        .def_readwrite("lambda_", &se::generation::MMRConfig::lambda);  // 'lambda' is Python keyword

    m.def("generate_candidates", &se::generation::generate_candidates,
          py::arg("history"), py::arg("number_weights") = Eigen::VectorXd{},
          py::arg("cfg") = se::generation::GeneratorConfig{});

    m.def("select_mmr", &se::generation::select_mmr,
          py::arg("candidates"), py::arg("cfg") = se::generation::MMRConfig{});

    m.def("strategy_name", &se::generation::strategy_name);
    m.def("jaccard_distance", &se::generation::jaccard_distance);

    // ---------- backtesting::walk_forward ----------
    py::class_<se::backtesting::WalkForwardConfig>(m, "WalkForwardConfig")
        .def(py::init<>())
        .def_readwrite("initial_train", &se::backtesting::WalkForwardConfig::initial_train)
        .def_readwrite("refit_every",   &se::backtesting::WalkForwardConfig::refit_every);

    py::class_<se::backtesting::ModelResult>(m, "ModelResult")
        .def_readonly("name",                &se::backtesting::ModelResult::name)
        .def_readonly("k_params",            &se::backtesting::ModelResult::k_params)
        .def_readonly("n_test",              &se::backtesting::ModelResult::n_test)
        .def_readonly("log_likelihood",      &se::backtesting::ModelResult::log_likelihood)
        .def_readonly("aic",                 &se::backtesting::ModelResult::aic)
        .def_readonly("bic",                 &se::backtesting::ModelResult::bic)
        .def_readonly("brier_score",         &se::backtesting::ModelResult::brier_score)
        .def_readonly("avg_hits_at_6",       &se::backtesting::ModelResult::avg_hits_at_6)
        .def_readonly("hits_std",            &se::backtesting::ModelResult::hits_std)
        .def_readonly("hits_distribution",   &se::backtesting::ModelResult::hits_distribution)
        .def_readonly("per_draw_hits",       &se::backtesting::ModelResult::per_draw_hits)
        .def_readonly("lift_vs_uniform_pct", &se::backtesting::ModelResult::lift_vs_uniform_pct);

    py::class_<se::backtesting::PermResult>(m, "PermResult")
        .def_readonly("delta_hits_vs_G0",    &se::backtesting::PermResult::delta_hits_vs_G0)
        .def_readonly("p_value",             &se::backtesting::PermResult::p_value)
        .def_readonly("robustly_beats_G0",   &se::backtesting::PermResult::robustly_beats_G0);

    py::class_<se::backtesting::WalkForward::FullReport>(m, "FullReport")
        .def_readonly("models",              &se::backtesting::WalkForward::FullReport::models)
        .def_readonly("permutation_tests",   &se::backtesting::WalkForward::FullReport::permutation_tests)
        .def_readonly("verdict",             &se::backtesting::WalkForward::FullReport::verdict)
        .def_readonly("robust_signal",       &se::backtesting::WalkForward::FullReport::robust_signal);

    py::class_<se::backtesting::WalkForward>(m, "WalkForward")
        .def(py::init<se::backtesting::WalkForwardConfig>(),
             py::arg("cfg") = se::backtesting::WalkForwardConfig{})
        .def_static("build_indicator", &se::backtesting::WalkForward::build_indicator)
        .def_static("build_regimes",   &se::backtesting::WalkForward::build_regimes)
        .def("qmat_G0", &se::backtesting::WalkForward::qmat_G0)
        .def("qmat_G2", &se::backtesting::WalkForward::qmat_G2, py::arg("draws"), py::arg("alpha") = 1.0)
        .def("qmat_G3", &se::backtesting::WalkForward::qmat_G3, py::arg("draws"),
             py::arg("lag") = 1, py::arg("alpha") = 1.0)
        .def("qmat_G4", &se::backtesting::WalkForward::qmat_G4)
        .def("qmat_G5", &se::backtesting::WalkForward::qmat_G5, py::arg("draws"), py::arg("alpha") = 1.0)
        .def("qmat_G6", &se::backtesting::WalkForward::qmat_G6,
             py::arg("g2"), py::arg("g3"), py::arg("g4"),
             py::arg("w") = std::array<double, 3>{0.5, 0.3, 0.2})
        .def("evaluate", &se::backtesting::WalkForward::evaluate)
        .def("paired_permutation_test", &se::backtesting::WalkForward::paired_permutation_test,
             py::arg("hits_a"), py::arg("hits_b"),
             py::arg("n_iter") = 4000, py::arg("seed") = 7)
        .def("run_all", &se::backtesting::WalkForward::run_all);
}
