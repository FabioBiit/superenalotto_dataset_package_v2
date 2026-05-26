# SuperEnalotto Engine (C++20)

High-performance reverse-engineering engine for the Italian SuperEnalotto.
Probabilistic modeling of the physical urn process — not a "break" of the
algorithm (there is no software algorithm: extraction is mechanical,
certified by ADM/GdF/Codacons).

## Scientific scope

| Module | Goal |
|---|---|
| `core/`         | CSV loader, PRNGs (PCG-XSL-RR-128, Philox-4x64-10, ChaCha20), statistical tests (chi2/KS/runs/Ljung-Box, SIMD), parallel Monte Carlo (OpenMP + CUDA) |
| `models/`       | Wallenius hypergeometric, Dirichlet-Multinomial MCMC, HMM regime (Baum-Welch/Viterbi), Hawkes self-exciting, PRNG fingerprinting |
| `inference/`    | NUTS/MH MCMC, posterior predictive checks, AIC/BIC/WAIC/LOO-CV |
| `generation/`   | Combination generation (6 strategies) + Maximal Marginal Relevance selector |
| `backtesting/`  | Walk-forward G0..G6 model evaluation + paired permutation tests |
| `python/`       | pybind11 bindings — exposes the engine as a Python module |

## Quickstart (Windows + MSVC + VS Community 2022)

Prerequisites (already verified on dev machine):
- Visual Studio 2022 Community with "Desktop development with C++"
- Python 3.11+ with `pybind11` installed (`pip install pybind11`)
- CMake 3.24+ and Ninja (auto-installable via `pip install cmake ninja`)
- CUDA Toolkit (OPTIONAL, only needed for `-Preset cuda`)

Dependencies (header/sources) are vendored under `cpp/deps/`:
- `deps/eigen/`  — Eigen 3.4 (header-only)
- `deps/catch2/` — Catch2 v3 (built as subdirectory)
- `pybind11`    — loaded via Python (`pip install pybind11`)

### Build

```powershell
cd cpp
.\scripts\build.ps1                       # CPU release + tests
.\scripts\build.ps1 -Preset cuda          # GPU acceleration (needs CUDA Toolkit)
.\scripts\build.ps1 -Configuration Debug
.\scripts\build.ps1 -SkipTests            # Skip test execution
```

The script automatically loads the MSVC environment from VS 2022 Community via
`vswhere.exe` + `vcvars64.bat`. No need to launch a "x64 Developer Command Prompt".

### Outputs

- `build/cpu/bin/Release/se_cli.exe`   — standalone CLI
- `build/cpu/lib/Release/se_engine.lib`— static library
- `build/cpu/lib/Release/se_engine.pyd`— Python extension module (pybind11)

## CLI commands

```bash
se_cli --version                                            # build info
se_cli --validate    data/.../full_history_validated.csv    # schema + range check
se_cli --frequency   data/.../full_history_validated.csv    # top-10 + chi-square
se_cli --fingerprint data/.../full_history_validated.csv    # test PCG64/Philox/ChaCha20
se_cli --generate    data/.../full_history_validated.csv 25 # 25 MMR combinations
se_cli --backtest    data/.../full_history_validated.csv 800 # walk-forward G0..G6
```

## Use from Python

After building, copy the `.pyd` to the project directory (or `pip install -e .`):

```python
import se_engine
draws = se_engine.load_csv("data/superenalotto_full_history_validated.csv")
freq  = se_engine.empirical_frequency(draws)

# Walk-forward backtesting
wf = se_engine.WalkForward(se_engine.WalkForwardConfig())
report = wf.run_all(draws)
for m in report.models:
    print(f"{m.name}: BIC={m.bic:.1f}, hits={m.avg_hits_at_6:.4f}, lift={m.lift_vs_uniform_pct:+.2f}%")
```

See `scripts/demo_from_python.py` for a complete example.

## Reference implementations

| Algorithm | Reference |
|---|---|
| PCG-XSL-RR-128 | O'Neill, *PCG: A Family of Simple Fast Space-Efficient...*, 2014 |
| Philox-4×64-10 | Salmon, Moraes, Dror, Shaw, *Parallel Random Numbers: As Easy as 1, 2, 3*, SC11, 2011 |
| ChaCha20       | Bernstein, *ChaCha, a variant of Salsa20*, 2008 |
| Wallenius      | Wallenius, *Biased Sampling: The Noncentral Hypergeometric...*, 1963 |
| Baum-Welch     | Baum, Petrie, Soules, Weiss (1970); Rabiner (1989) tutorial |
| Hawkes         | Hawkes (1971) self-exciting point processes; Ogata (1981) ML |

## Constraints (from project CLAUDE.md)

- Only public historical data; no infrastructure access.
- Models reverse-engineer the **statistical distribution** of outputs, not the
  physical mechanism.
- Every combination has theoretical probability **1 / 622,614,630** under uniform iid.
- Never present output as a prediction.
