# SuperEnalotto Engine (C++20)

High-performance reverse-engineering engine for the Italian SuperEnalotto.
Probabilistic modeling of the physical urn process — not a "break" of the algorithm
(there is no software algorithm: extraction is mechanical, certified by ADM/GdF/Codacons).

## Scientific scope

| Module | Goal |
|---|---|
| `core/`       | CSV loader, PRNGs (PCG64/Philox/ChaCha20), stat tests (SIMD), parallel Monte Carlo |
| `models/`     | Wallenius hypergeometric, Dirichlet-Multinomial MCMC, HMM regime, Hawkes, PRNG fingerprint |
| `inference/`  | NUTS/MH MCMC, posterior predictive checks, AIC/BIC/WAIC/LOO-CV |
| `generation/` | Combination generation strategies + MMR portfolio selector |
| `python/`     | pybind11 bindings — exposes the engine as a Python module |

## Build

```bash
# Prereq: vcpkg installed, VCPKG_ROOT env var set, CMake 3.24+, Ninja
cd cpp
cmake --preset cpu          # CPU-only build
cmake --build --preset cpu-release

# Or with CUDA (requires NVIDIA GPU + CUDA Toolkit)
cmake --preset cuda
cmake --build --preset cuda-release
```

## Output

- `build/<preset>/bin/se_cli` — standalone CLI
- `build/<preset>/lib/libse_engine.*` — static library
- `build/<preset>/lib/se_engine_py.<ext>` — Python module (pybind11)

## Constraints (from project CLAUDE.md)

- Only public historical data; no infrastructure access.
- Models reverse-engineer the **statistical distribution** of outputs, not the
  physical mechanism.
- Every combination has theoretical probability **1 / 622,614,630** under uniform iid.
- Never present output as a prediction.
