"""
Generate 100 ranked SuperEnalotto combinations using the C++ engine via pybind11.
Outputs:
    inference/phaseC100_combinations.csv
    inference/phaseC100_summary.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

import se_engine  # type: ignore

CSV_IN  = ROOT / "data" / "superenalotto_full_history_validated.csv"
CSV_OUT = ROOT / "inference" / "phaseC100_combinations.csv"
JSON_OUT = ROOT / "inference" / "phaseC100_summary.json"

WARNING = (
    "Le combinazioni generate non sono previsioni. Sono il risultato di criteri "
    "statistici, euristici, machine learning e teoria delle decisioni applicati "
    "su dati storici. Nel SuperEnalotto ogni combinazione ha esattamente la stessa "
    "probabilita teorica di essere estratta se il processo e uniforme e indipendente."
)

N_PER_STRAT = 20         # 20 * 6 = 120 candidati (margine per dedup/filtri qualita)
K_FINAL     = 100
MMR_LAMBDA  = 0.65
SEED        = 42


def main():
    print(f"se_engine v{se_engine.__version__}")
    draws = se_engine.load_csv(str(CSV_IN))
    print(f"Loaded {len(draws)} draws from {CSV_IN.name}")

    gen_cfg = se_engine.GeneratorConfig()
    gen_cfg.n_per_strategy = N_PER_STRAT
    gen_cfg.seed = SEED
    gen_cfg.overlap_window = 10

    mmr_cfg = se_engine.MMRConfig()
    mmr_cfg.k_final = K_FINAL
    mmr_cfg.lambda_ = MMR_LAMBDA

    candidates = se_engine.generate_candidates(draws, [], gen_cfg)
    final = se_engine.select_mmr(candidates, mmr_cfg)
    print(f"Generated {len(candidates)} candidates, MMR selected {len(final)}")

    print()
    print(WARNING)
    print()

    # Strategy distribution
    strat_count = {}
    for c in final:
        s = se_engine.strategy_name(c.strategy)
        strat_count[s] = strat_count.get(s, 0) + 1
    print("Strategy distribution:")
    for s, n in sorted(strat_count.items(), key=lambda x: -x[1]):
        print(f"  {s:24s} {n:3d}")
    print()

    # CSV
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# WARNING: {WARNING}"])
        w.writerow(["rank", "strategy", "combinazione", "jolly", "superstar",
                    "score", "n_even", "n_decadi"])
        for i, c in enumerate(final, 1):
            combo = "-".join(f"{n:02d}" for n in c.main)
            n_even = sum(1 for n in c.main if n % 2 == 0)
            n_dec  = len({(n - 1) // 10 for n in c.main})
            w.writerow([i, se_engine.strategy_name(c.strategy),
                        combo, c.jolly, c.superstar,
                        round(c.score, 2), n_even, n_dec])
    print(f"CSV  -> {CSV_OUT.relative_to(ROOT)}")

    # JSON summary
    summary = {
        "engine": f"se_engine v{se_engine.__version__} (C++)",
        "dataset": CSV_IN.name,
        "n_draws": len(draws),
        "params": {
            "n_per_strategy": N_PER_STRAT, "k_final": K_FINAL,
            "mmr_lambda": MMR_LAMBDA, "seed": SEED,
        },
        "n_candidates_generated": len(candidates),
        "n_in_portfolio": len(final),
        "strategy_distribution": strat_count,
        "warning": WARNING,
        "combinations": [{
            "rank": i,
            "strategy": se_engine.strategy_name(c.strategy),
            "main": list(c.main),
            "jolly": c.jolly,
            "superstar": c.superstar,
            "score": round(c.score, 2),
        } for i, c in enumerate(final, 1)],
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON -> {JSON_OUT.relative_to(ROOT)}")

    # Display top 20
    print()
    print("Top 20:")
    print(f"  {'#':>3} {'strategy':24s} {'combinazione':22s} {'J':>3} {'SS':>3} {'score':>6}")
    for i, c in enumerate(final[:20], 1):
        combo = "-".join(f"{n:02d}" for n in c.main)
        print(f"  {i:3d} {se_engine.strategy_name(c.strategy):24s} {combo:22s} "
              f"{c.jolly:3d} {c.superstar:3d} {c.score:6.1f}")
    print(f"  ... ({len(final)-20} more in CSV)")


if __name__ == "__main__":
    main()
