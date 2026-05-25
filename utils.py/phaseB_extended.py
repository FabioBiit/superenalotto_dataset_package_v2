"""
Phase B - Extended dataset (2009-2026, 2855 draws)
==================================================
Same model hierarchy G0..G6 as utils.py/generator_hierarchy.py, run on the
validated full-history CSV. Detection threshold drops from ~48% (1133 draws)
to ~25% per-number bias (2855 draws), making weak signals easier to spot.

Output:
    inference/phaseB_ext_model_selection.csv
    inference/phaseB_ext_permutation_test.csv
    inference/phaseB_ext_ppc.json
    inference/phaseB_ext_summary.json
    inference/phaseB_comparison_old_vs_extended.csv
"""
from __future__ import annotations
import json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

BASE = Path(r"C:\Users\kyros\OneDrive\Desktop\superenalotto_dataset_package_v2\superenalotto_dataset_package_v2")
CSV = BASE / "data" / "superenalotto_full_history_validated.csv"
OUT = BASE / "inference"
OUT.mkdir(exist_ok=True)

INITIAL_TRAIN = 800   # bigger train window for 2855-draw dataset
REFIT_EVERY   = 100
PPC_SIMS_PER_DRAW = 1

CP_FLUTTER = pd.Timestamp("2022-08-04")
CP_4WEEK   = pd.Timestamp("2023-07-07")

COLS = ["date", "contest_number", "n1", "n2", "n3", "n4", "n5", "n6", "jolly", "superstar"]
df = pd.read_csv(CSV, usecols=COLS)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "contest_number"]).reset_index(drop=True)
MAIN = ["n1", "n2", "n3", "n4", "n5", "n6"]
N = len(df)

draw_mat = np.zeros((N, 90), dtype=np.int8)
for i in range(N):
    for c in MAIN:
        draw_mat[i, int(df.iloc[i][c]) - 1] = 1
assert (draw_mat.sum(axis=1) == 6).all()
print(f"[LOAD] N={N} draws  {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

regimes = np.zeros(N, dtype=int)
for i in range(N):
    d = df.iloc[i]["date"]
    regimes[i] = 0 if d < CP_FLUTTER else (1 if d < CP_4WEEK else 2)
print(f"[REGIME] pre-Flutter={int((regimes==0).sum())} "
      f"Flutter={int((regimes==1).sum())} 4-week={int((regimes==2).sum())}")

draws_obs = []
for i in range(N):
    idx = np.where(draw_mat[i])[0] + 1
    draws_obs.append(tuple(sorted(int(x) for x in idx)))


def summary_stats(draws_list):
    n = len(draws_list)
    nums_flat = np.array([x for d in draws_list for x in d])
    freq = np.bincount(nums_flat, minlength=91)[1:91]
    exp_per = n * 6 / 90
    chi2 = float(((freq - exp_per) ** 2 / exp_per).sum())
    sums = np.array([sum(d) for d in draws_list])
    rngs = np.array([max(d) - min(d) for d in draws_list])
    evens = np.array([sum(1 for x in d if x % 2 == 0) for d in draws_list])
    decades = np.array([len(set((x - 1) // 10 for x in d)) for d in draws_list])
    consec_pairs = [sum(1 for i in range(1, 6) if sorted(d)[i] == sorted(d)[i - 1] + 1) for d in draws_list]
    consec = np.array(consec_pairs)
    return {"chi2_freq": chi2,
            "sum_mean": float(sums.mean()), "sum_std": float(sums.std()),
            "range_mean": float(rngs.mean()),
            "evens_mean": float(evens.mean()),
            "decades_mean": float(decades.mean()),
            "consec_mean": float(consec.mean())}


obs_stats_all = summary_stats(draws_obs)
obs_stats_test = summary_stats(draws_obs[INITIAL_TRAIN:])
print(f"[OBS-ALL]  {obs_stats_all}")
print(f"[OBS-TEST] {obs_stats_test}")


def qmat_G0():
    return np.full((N, 90), 6.0 / 90.0)

def qmat_G2(alpha=1.0):
    q = np.empty((N, 90))
    cum = np.zeros(90)
    for t in range(N):
        total = cum.sum() + 90 * alpha
        q[t] = (cum + alpha) / total * 6.0
        cum += draw_mat[t]
    return q

def qmat_G3(lag=1, alpha=1.0):
    q = np.full((N, 90), 6.0 / 90.0)
    start = max(lag, INITIAL_TRAIN)
    for t in range(start, N):
        # Vectorized: build transition counts up to t-1 for ALL 90 numbers at once
        prev = draw_mat[lag - 1:t - 1]   # (t-lag, 90)
        curr = draw_mat[lag:t]            # same
        n11 = ((prev == 1) & (curr == 1)).sum(axis=0).astype(float)
        n10 = ((prev == 1) & (curr == 0)).sum(axis=0).astype(float)
        n01 = ((prev == 0) & (curr == 1)).sum(axis=0).astype(float)
        n00 = ((prev == 0) & (curr == 0)).sum(axis=0).astype(float)
        prev_at_t = draw_mat[t - 1]  # (90,)
        p_if_1 = (n11 + alpha) / (n11 + n10 + 2 * alpha)
        p_if_0 = (n01 + alpha) / (n01 + n00 + 2 * alpha)
        p = np.where(prev_at_t == 1, p_if_1, p_if_0)
        s = p.sum()
        if s > 0:
            q[t] = p / s * 6.0
    return q

def qmat_G4(refit_every=REFIT_EVERY):
    q = np.full((N, 90), 6.0 / 90.0)
    nums = np.arange(1, 91)
    decade = (nums - 1) // 10
    parity = nums % 2
    fascia = np.where(nums <= 30, 0, np.where(nums <= 60, 1, 2))
    dec_d = np.eye(9)[decade]
    par_d = np.column_stack([1 - parity, parity])
    fas_d = np.eye(3)[fascia]
    X = np.column_stack([dec_d, par_d, fas_d, np.ones(90)])
    last_refit = -1
    beta = None
    for t in range(INITIAL_TRAIN, N):
        if (t - last_refit) >= refit_every:
            counts = draw_mat[:t].sum(axis=0).astype(float)
            p_emp = (counts + 1) / (counts.sum() + 90)
            logit = np.log(p_emp / (1 - p_emp))
            beta, *_ = np.linalg.lstsq(X, logit, rcond=None)
            last_refit = t
        logit_pred = X @ beta
        p = 1.0 / (1.0 + np.exp(-logit_pred))
        p = p / p.sum() * 6.0
        q[t] = np.clip(p, 1e-6, 1 - 1e-6)
    return q

def qmat_G5(alpha=1.0):
    q = np.full((N, 90), 6.0 / 90.0)
    for t in range(1, N):
        reg = regimes[t]
        same_reg_past = (regimes[:t] == reg)
        if same_reg_past.sum() < 30:
            counts = draw_mat[:t].sum(axis=0).astype(float)
        else:
            counts = draw_mat[:t][same_reg_past].sum(axis=0).astype(float)
        total = counts.sum() + 90 * alpha
        q[t] = (counts + alpha) / total * 6.0
    return q

def qmat_G6(q2, q3, q4, w=(0.5, 0.3, 0.2)):
    return w[0] * q2 + w[1] * q3 + w[2] * q4


def evaluate(name, q, k_params, train_start=INITIAL_TRAIN):
    eps = 1e-9
    q_e = np.clip(q[train_start:], eps, 1 - eps)
    y_e = draw_mat[train_start:]
    n_cells = y_e.size
    ll = float(np.sum(y_e * np.log(q_e) + (1 - y_e) * np.log(1 - q_e)))
    brier = float(np.mean((q_e - y_e) ** 2))
    aic = 2 * k_params - 2 * ll
    bic = k_params * math.log(n_cells) - 2 * ll
    hits = []
    for ti in range(q_e.shape[0]):
        actual_idx = np.where(y_e[ti])[0]
        top6 = np.argsort(-q_e[ti])[:6]
        hits.append(int(np.isin(top6, actual_idx).sum()))
    hits = np.array(hits)
    hd = {f"hit_{k}": int((hits == k).sum()) for k in range(7)}
    return {"model": name, "k_params": k_params,
            "log_likelihood": ll, "AIC": aic, "BIC": bic,
            "brier_score": brier,
            "avg_hits_at_6": float(hits.mean()),
            "hits_std": float(hits.std()),
            **hd}


def simulate_from_q(q, train_start=INITIAL_TRAIN, seed=123):
    rng = np.random.default_rng(seed)
    sims = []
    for t in range(train_start, N):
        p = q[t].astype(float).copy()
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        try:
            picks = rng.choice(90, size=6, replace=False, p=p)
        except Exception:
            picks = rng.choice(90, size=6, replace=False)
        sims.append(tuple(sorted(int(x) + 1 for x in picks)))
    return sims


print("\n[FIT] G0 (uniform) ...")
q_G0 = qmat_G0()
print("[FIT] G2 (Dirichlet-Multinomial) ...")
q_G2 = qmat_G2(alpha=1.0)
print("[FIT] G3 (Markov lag-1, vectorized) ...")
q_G3 = qmat_G3(lag=1)
print("[FIT] G4 (structural GLM) ...")
q_G4 = qmat_G4()
print("[FIT] G5 (regime-split G2) ...")
q_G5 = qmat_G5(alpha=1.0)
print("[FIT] G6 (ensemble G2+G3+G4) ...")
q_G6 = qmat_G6(q_G2, q_G3, q_G4)

print("\n[EVAL]")
results = [
    evaluate("G0_uniform",      q_G0, k_params=0),
    evaluate("G2_dirichlet",    q_G2, k_params=90),
    evaluate("G3_markov_lag1",  q_G3, k_params=180),
    evaluate("G4_structural",   q_G4, k_params=14),
    evaluate("G5_regime_split", q_G5, k_params=270),
    evaluate("G6_ensemble",     q_G6, k_params=286),
]
res_df = pd.DataFrame(results)
uniform_hits = float(res_df[res_df["model"] == "G0_uniform"]["avg_hits_at_6"].iloc[0])
res_df["lift_vs_uniform_pct"] = (res_df["avg_hits_at_6"] / uniform_hits - 1) * 100
res_df = res_df.sort_values("BIC").reset_index(drop=True)
print(res_df[["model", "k_params", "log_likelihood", "AIC", "BIC",
              "brier_score", "avg_hits_at_6", "lift_vs_uniform_pct"]].to_string(index=False))

print("\n[PPC] simulating and comparing summary stats ...")
ppc_results = {}
for name, q in [("G0", q_G0), ("G2", q_G2), ("G3", q_G3), ("G4", q_G4), ("G5", q_G5), ("G6", q_G6)]:
    sim = simulate_from_q(q)
    s = summary_stats(sim)
    diffs = {k: float(s[k] - obs_stats_test[k]) for k in obs_stats_test}
    ppc_results[name] = {"sim_stats": s, "diffs_vs_obs": diffs}
    print(f"  {name}: chi2_diff={diffs['chi2_freq']:+.2f}  "
          f"sum_diff={diffs['sum_mean']:+.3f}  "
          f"decades_diff={diffs['decades_mean']:+.3f}  "
          f"consec_diff={diffs['consec_mean']:+.3f}")

print("\n[G1] PRNG references (all uniform, sanity check):")
for prng_name, bitgen in [("MT19937", np.random.MT19937), ("PCG64", np.random.PCG64), ("Philox", np.random.Philox)]:
    rng = np.random.Generator(bitgen(42))
    sim = [tuple(sorted(rng.choice(90, size=6, replace=False) + 1)) for _ in range(N - INITIAL_TRAIN)]
    s = summary_stats(sim)
    diffs = {k: float(s[k] - obs_stats_test[k]) for k in obs_stats_test}
    ppc_results[f"G1_{prng_name}"] = {"sim_stats": s, "diffs_vs_obs": diffs}
    print(f"  {prng_name}: chi2_diff={diffs['chi2_freq']:+.2f}  "
          f"sum_diff={diffs['sum_mean']:+.3f}  "
          f"decades_diff={diffs['decades_mean']:+.3f}")


def paired_perm(arr_a, arr_b, iters=5000, seed=7):
    rng = np.random.default_rng(seed)
    diff = arr_a - arr_b
    obs = diff.mean()
    boot = np.empty(iters)
    signs = rng.choice([-1, 1], size=(iters, len(diff)))
    for k in range(iters):
        boot[k] = (diff * signs[k]).mean()
    return obs, float((np.abs(boot) >= abs(obs)).mean())


def hits_array(q):
    arr = []
    for ti in range(INITIAL_TRAIN, N):
        actual_idx = np.where(draw_mat[ti])[0]
        top6 = np.argsort(-q[ti])[:6]
        arr.append(int(np.isin(top6, actual_idx).sum()))
    return np.array(arr)


h0 = hits_array(q_G0)
qmap = {"G2_dirichlet": q_G2, "G3_markov_lag1": q_G3, "G4_structural": q_G4,
        "G5_regime_split": q_G5, "G6_ensemble": q_G6}
perm_rows = []
for name, q in qmap.items():
    hx = hits_array(q)
    obs, pv = paired_perm(hx, h0, iters=4000)
    perm_rows.append({"model": name, "delta_hits_vs_G0": round(float(obs), 4),
                      "perm_p_value": round(pv, 4),
                      "robustly_beats_G0": bool(pv < 0.05 and obs > 0)})
perm_df = pd.DataFrame(perm_rows).sort_values("delta_hits_vs_G0", ascending=False).reset_index(drop=True)
print("\n[PERM TEST] models vs G0:")
print(perm_df.to_string(index=False))

best_bic_row = res_df.iloc[0]
best_hits_row = res_df.sort_values("avg_hits_at_6", ascending=False).iloc[0]
lift_best = float(best_hits_row["lift_vs_uniform_pct"])
robust_signal = bool((lift_best > 5) and any(perm_df["robustly_beats_G0"]))
verdict = {
    "best_by_BIC": best_bic_row["model"],
    "best_by_hits": best_hits_row["model"],
    "G0_avg_hits": round(uniform_hits, 4),
    "best_avg_hits": round(float(best_hits_row["avg_hits_at_6"]), 4),
    "lift_best_pct": round(lift_best, 3),
    "robust_signal": robust_signal,
    "interpretation": (
        "Dati compatibili con uniforme. Nessun modello batte G0 in modo robusto. "
        "Le combinazioni generate restano euristiche (non previsioni)."
        if not robust_signal else
        f"Modello {best_hits_row['model']} mostra lift {lift_best:.2f}% statisticamente "
        "rilevante. Verificare con out-of-sample/walk-forward esteso."
    ),
}
print(f"\n[VERDICT] {json.dumps(verdict, indent=2, ensure_ascii=False)}")

# Save
res_df.to_csv(OUT / "phaseB_ext_model_selection.csv", index=False)
perm_df.to_csv(OUT / "phaseB_ext_permutation_test.csv", index=False)
with open(OUT / "phaseB_ext_ppc.json", "w", encoding="utf-8") as f:
    json.dump({"obs_stats_test": obs_stats_test, "models": ppc_results}, f, indent=2, ensure_ascii=False)
with open(OUT / "phaseB_ext_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_extractions": N,
        "date_min": str(df["date"].iloc[0].date()),
        "date_max": str(df["date"].iloc[-1].date()),
        "initial_train": INITIAL_TRAIN,
        "regime_counts": {
            "pre_Flutter": int((regimes == 0).sum()),
            "Flutter": int((regimes == 1).sum()),
            "4_week_post_2023_07_07": int((regimes == 2).sum()),
        },
        "obs_stats_all": obs_stats_all,
        "obs_stats_test": obs_stats_test,
        "model_eval": res_df.to_dict(orient="records"),
        "permutation_test": perm_df.to_dict(orient="records"),
        "verdict": verdict,
    }, f, indent=2, ensure_ascii=False, default=str)

# Side-by-side comparison with original Phase B
try:
    old = pd.read_csv(OUT / "phaseB_model_selection.csv")
    old["dataset"] = "1133_draws"
    new = res_df.copy()
    new["dataset"] = "2855_draws"
    comp_cols = ["dataset", "model", "log_likelihood", "AIC", "BIC", "brier_score",
                 "avg_hits_at_6", "lift_vs_uniform_pct"]
    comp = pd.concat([old[comp_cols], new[comp_cols]], ignore_index=True)
    comp = comp.sort_values(["model", "dataset"]).reset_index(drop=True)
    comp.to_csv(OUT / "phaseB_comparison_old_vs_extended.csv", index=False)
    print(f"\n[COMP] saved to {OUT / 'phaseB_comparison_old_vs_extended.csv'}")
except Exception as e:
    print(f"[WARN] comparison skipped: {e}")

print(f"\n[OUT] saved to {OUT}/")
print("[DONE] Phase B Extended finished.")
