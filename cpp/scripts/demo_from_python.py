"""
Demo: how to use the C++ SuperEnalotto Engine from Python via pybind11.

Prerequisites (after building):
    The compiled module 'se_engine' must be in PYTHONPATH or installed in your venv.

    Typical workflow:
        cd cpp
        cmake --preset cpu && cmake --build --preset cpu-release
        cp build/cpu/lib/Release/se_engine*.pyd .   # or .so on Linux
        python scripts/demo_from_python.py
"""
import sys
from pathlib import Path

try:
    import se_engine
except ImportError:
    sys.exit("se_engine module not found - build the C++ project first")


CSV = Path(__file__).resolve().parents[2] / "data" / "superenalotto_full_history_validated.csv"


def main():
    print(f"se_engine v{se_engine.__version__}")
    print(f"C(6,90) = {se_engine.C_6_90:,}\n")

    draws = se_engine.load_csv(str(CSV))
    print(f"Loaded {len(draws)} draws from {CSV.name}")
    print(f"Invalid: {se_engine.count_invalid(draws)}\n")

    freq = se_engine.empirical_frequency(draws)
    print("Top-10 by frequency:")
    sorted_freq = sorted(enumerate(freq, start=1), key=lambda x: -x[1])
    for n, f in sorted_freq[:10]:
        print(f"  {n:2d}  {f:.5f}")

    print("\nPRNG fingerprint comparison:")
    fp = se_engine.compare_prngs_against_observed(draws, ["PCG64", "Philox", "ChaCha20"], 42)
    for r in fp:
        print(f"  {r.prng_name:10s} chi2_p={r.chi_square.p_value:.4f} "
              f"ks_p={r.ks_test.p_value:.4f} JS={r.js_divergence_vs_observed:.5f} "
              f"score={r.overall_score:.3f}")

    print("\nMonte Carlo simulation (1M uniform draws):")
    mc_cfg = se_engine.MCConfig()
    mc_cfg.n_simulations = 1_000_000
    mc_cfg.seed = 42
    sim_freq = se_engine.simulate_uniform_frequency(1_000_000, mc_cfg)
    print(f"  Simulated frequency: min={sim_freq.min():.6f}  max={sim_freq.max():.6f}")
    print(f"  Theoretical uniform: {1.0/90:.6f}")

    print("\nGenerate 25 candidates via MMR portfolio:")
    gen_cfg = se_engine.GeneratorConfig()
    gen_cfg.n_per_strategy = 8
    gen_cfg.seed = 42
    candidates = se_engine.generate_candidates(draws, [], gen_cfg)
    print(f"  Generated {len(candidates)} candidates")

    mmr_cfg = se_engine.MMRConfig()
    mmr_cfg.k_final = 25
    mmr_cfg.lambda_ = 0.65
    final = se_engine.select_mmr(candidates, mmr_cfg)
    print(f"\nWARNING: Le combinazioni generate non sono previsioni. "
          f"Nel SuperEnalotto ogni combinazione ha esattamente la stessa "
          f"probabilita teorica di 1/{se_engine.C_6_90:,}.\n")
    print(f"{'Rank':<5}{'Strategy':<24}{'Combination':<28}{'J':<5}{'SS':<5}{'Score':<6}")
    for i, c in enumerate(final, start=1):
        combo = "-".join(f"{x:02d}" for x in c.main)
        strat = se_engine.strategy_name(c.strategy)
        print(f"{i:<5}{strat:<24}{combo:<28}{c.jolly:<5}{c.superstar:<5}{c.score:<6.1f}")


if __name__ == "__main__":
    main()
