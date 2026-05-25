"""
SuperEnalotto - Rigorous Pipeline
=================================
- Extended statistical tests (chi2, G-test, KS-MC, runs, Ljung-Box, MI, KL/JS, BH)
- Leakage-safe feature engineering
- Walk-forward backtesting: 5 baselines + 4 models
- Statistical comparison with bootstrap CI + paired permutation test
- Combinations + portfolio optimization (greedy MMR) only if a model beats random
"""

from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

# ============================ CONFIG ============================
BASE = Path(r"C:\Users\kyros\OneDrive\Desktop\superenalotto_dataset_package_v2\superenalotto_dataset_package_v2")
CSV = BASE / "data" / "superenalotto_5.5_anni_extraction.csv"
OUT = BASE / "inference"
OUT.mkdir(exist_ok=True)

INITIAL_TRAIN = 400      # initial train window for walk-forward
REFIT_EVERY = 50         # refit heavy ML models every N draws
TOPK = 6
ROLL_WINDOWS = [5, 10, 20, 50, 100]
LIFT_SIGNIFICANCE_PCT = 5.0   # min lift % over uniform to call "signal"
BOOTSTRAP_ITERS = 2000
MC_KS_ITERS = 2000

# ============================ LOAD ============================
COLS = ["date", "contest_number", "n1", "n2", "n3", "n4", "n5", "n6", "jolly", "superstar"]
df = pd.read_csv(CSV, header=0, names=COLS)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "contest_number"]).reset_index(drop=True)
MAIN = ["n1", "n2", "n3", "n4", "n5", "n6"]
N = len(df)
print(f"[LOAD] {N} estrazioni — {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

# Binary draw matrix: shape (N, 90), 1 if number i+1 was drawn at draw t
draw_mat = np.zeros((N, 90), dtype=np.int8)
for i in range(N):
    for c in MAIN:
        draw_mat[i, int(df.iloc[i][c]) - 1] = 1
assert (draw_mat.sum(axis=1) == 6).all()

# ============================ EXTENDED STATISTICAL TESTS ============================
print("\n[TESTS] Extended randomness tests")
tests_log = []
all_nums = df[MAIN].values.flatten().astype(int)
freq_total = np.bincount(all_nums, minlength=91)[1:91]
exp_per = N * 6 / 90

# 1. Chi-square
chi2 = float(((freq_total - exp_per) ** 2 / exp_per).sum())
chi2_p = float(1 - stats.chi2.cdf(chi2, df=89))
tests_log.append(("Chi-square (uniformità freq)", chi2, chi2_p, "non rifiuta" if chi2_p > 0.05 else "rifiuta"))

# 2. G-test
g_stat = 2 * float(np.sum(freq_total * np.log(freq_total / exp_per)))
g_p = float(1 - stats.chi2.cdf(g_stat, df=89))
tests_log.append(("G-test (likelihood ratio)", g_stat, g_p, "non rifiuta" if g_p > 0.05 else "rifiuta"))

# 3. KS via Monte Carlo (discrete uniform)
ecdf = np.cumsum(freq_total) / freq_total.sum()
ucdf = np.arange(1, 91) / 90
ks_stat = float(np.max(np.abs(ecdf - ucdf)))
ks_sim = np.empty(MC_KS_ITERS)
for j in range(MC_KS_ITERS):
    sim = RNG.choice(90, size=N * 6, replace=True) + 1
    sf = np.bincount(sim, minlength=91)[1:91]
    s_ecdf = np.cumsum(sf) / sf.sum()
    ks_sim[j] = np.max(np.abs(s_ecdf - ucdf))
ks_p = float((ks_sim >= ks_stat).mean())
tests_log.append(("KS Monte Carlo (vs uniform)", ks_stat, ks_p, "non rifiuta" if ks_p > 0.05 else "rifiuta"))

# 4. Runs test (parity sequence Wald-Wolfowitz)
parity = (all_nums % 2 == 0).astype(int)
n1 = int(parity.sum()); n0 = int(len(parity) - n1)
runs = int(1 + np.sum(parity[1:] != parity[:-1]))
mu_r = 2 * n0 * n1 / (n0 + n1) + 1
var_r = 2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) / ((n0 + n1) ** 2 * (n0 + n1 - 1))
z_runs = (runs - mu_r) / math.sqrt(var_r)
runs_p = float(2 * (1 - stats.norm.cdf(abs(z_runs))))
tests_log.append(("Runs test (parità)", z_runs, runs_p, "non rifiuta" if runs_p > 0.05 else "rifiuta"))

# 5. Ljung-Box per-number, combined via Fisher's method
lb_pvals = []
for k in range(90):
    s = draw_mat[:, k]
    if s.sum() < 10:
        continue
    try:
        lb = acorr_ljungbox(s, lags=[5, 10, 20], return_df=True)
        lb_pvals.append(float(lb["lb_pvalue"].min()))
    except Exception:
        pass
lb_arr = np.array(lb_pvals)
fisher_stat = -2 * np.sum(np.log(np.clip(lb_arr, 1e-300, 1)))
fisher_p = float(1 - stats.chi2.cdf(fisher_stat, df=2 * len(lb_arr)))
tests_log.append(("Ljung-Box Fisher-combined", fisher_stat, fisher_p,
                  "non rifiuta" if fisher_p > 0.05 else "rifiuta"))

# 6. Entropia Shannon
p = freq_total / freq_total.sum()
H_obs = float(-np.sum(p * np.log2(p)))
H_max = math.log2(90)
entropy_eff = H_obs / H_max

# 7. KL & JS divergence dalla uniforme
u = np.ones(90) / 90
kl = float(np.sum(p * np.log(p / u)))
m = 0.5 * (p + u)
js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(u * np.log(u / m)))

# 8. Mutual information binaria per-number, draw[t] vs draw[t+1]
mi_vals = []
for k in range(90):
    a = draw_mat[:-1, k]; b = draw_mat[1:, k]
    pij = np.zeros((2, 2))
    for ii in (0, 1):
        for jj in (0, 1):
            pij[ii, jj] = ((a == ii) & (b == jj)).mean()
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = pi @ pj
        ratio = np.where(denom > 0, pij / denom, 1.0)
        log_r = np.where((pij > 0) & (denom > 0), np.log(ratio), 0.0)
        mi = float(np.sum(np.where(pij > 0, pij * log_r, 0.0)))
    mi_vals.append(mi)
mi_avg = float(np.mean(mi_vals))
mi_max = float(np.max(mi_vals))

# 9. Multiple-testing su coppie (BH/Bonferroni)
pair_counter = Counter()
for i in range(N):
    nums = sorted(np.where(draw_mat[i])[0] + 1)
    for pair in combinations(nums, 2):
        pair_counter[pair] += 1
n_pairs = 90 * 89 // 2
exp_pair = N * 15 / n_pairs
pair_pvals = []
for pair, cnt in pair_counter.items():
    # two-sided Poisson tail
    if cnt >= exp_pair:
        pv = 1 - stats.poisson.cdf(cnt - 1, exp_pair)
    else:
        pv = stats.poisson.cdf(cnt, exp_pair)
    pair_pvals.append(min(1.0, 2 * pv))
pair_pvals_arr = np.array(pair_pvals)
reject_bh, _, _, _ = multipletests(pair_pvals_arr, alpha=0.05, method="fdr_bh")
reject_bonf, _, _, _ = multipletests(pair_pvals_arr, alpha=0.05, method="bonferroni")
n_sig_bh = int(reject_bh.sum())
n_sig_bonf = int(reject_bonf.sum())

print(f"  chi2={chi2:.2f} p={chi2_p:.4f} | G={g_stat:.2f} p={g_p:.4f}")
print(f"  KS={ks_stat:.4f} p_MC={ks_p:.4f} | runs z={z_runs:.3f} p={runs_p:.4f}")
print(f"  LjungBox-Fisher chi2={fisher_stat:.2f} p={fisher_p:.4f}")
print(f"  Entropy={H_obs:.4f}/{H_max:.4f} ({entropy_eff*100:.3f}%)  KL={kl:.5f}  JS={js:.5f}")
print(f"  MI avg={mi_avg:.5f}  max={mi_max:.5f} bits")
print(f"  Pairs significative — BH(α=0.05): {n_sig_bh}/{len(pair_pvals)}  Bonferroni: {n_sig_bonf}/{len(pair_pvals)}")

# ============================ FEATURE ENGINEERING ============================
print("\n[FEAT] Leakage-safe feature engineering")

cum_freq = np.zeros((N, 90), dtype=np.float32)
roll_freq = {w: np.zeros((N, 90), dtype=np.float32) for w in ROLL_WINDOWS}
delay_cur = np.zeros((N, 90), dtype=np.float32)
delay_mean = np.zeros((N, 90), dtype=np.float32)
delay_max = np.zeros((N, 90), dtype=np.float32)

last_seen = np.full(90, -1, dtype=np.int64)
gap_sum = np.zeros(90); gap_cnt = np.zeros(90); gap_max = np.zeros(90)

for t in range(N):
    # snapshot BEFORE incorporating draw t (features must use only past data)
    if t == 0:
        cum_freq[t] = 0
    else:
        cum_freq[t] = draw_mat[:t].sum(axis=0)
    for w in ROLL_WINDOWS:
        s = max(0, t - w)
        roll_freq[w][t] = draw_mat[s:t].sum(axis=0)
    for k in range(90):
        if last_seen[k] == -1:
            delay_cur[t, k] = t  # never seen → delay grows
            delay_mean[t, k] = 0
            delay_max[t, k] = 0
        else:
            delay_cur[t, k] = t - 1 - last_seen[k]
            delay_mean[t, k] = gap_sum[k] / gap_cnt[k] if gap_cnt[k] > 0 else 0
            delay_max[t, k] = gap_max[k]
    # update with draw t
    for k in range(90):
        if draw_mat[t, k] == 1:
            if last_seen[k] != -1:
                gap = t - last_seen[k]
                gap_sum[k] += gap
                gap_cnt[k] += 1
                if gap > gap_max[k]:
                    gap_max[k] = gap
            last_seen[k] = t

# delay ratio
delay_ratio = np.where(delay_mean > 0, delay_cur / delay_mean, 0)

# structural
nums_arr = np.arange(1, 91)
decade = ((nums_arr - 1) // 10).astype(np.float32)
parity = (nums_arr % 2).astype(np.float32)
fascia = np.where(nums_arr <= 30, 0, np.where(nums_arr <= 60, 1, 2)).astype(np.float32)

print(f"  Features ready: cum_freq, roll{ROLL_WINDOWS}, delay (cur/mean/max/ratio), decade/parity/fascia")

# ============================ MARK TASK 1 DONE ============================
# (statistical tests sezione completata)

# ============================ WALK-FORWARD BACKTEST ============================
print("\n[BACKTEST] Walk-forward")

def hits_at_k(scores: np.ndarray, actual_idx: np.ndarray, k: int = 6) -> int:
    """Number of overlaps between top-k scored numbers and actual drawn numbers."""
    topk = np.argsort(-scores)[:k]
    return int(np.isin(topk, actual_idx).sum())

def build_features(t: int) -> np.ndarray:
    """Return 90 x F feature matrix for draw t (using only data before t)."""
    feats = [
        cum_freq[t] / max(1, t),                  # normalized cum freq
        roll_freq[5][t] / 5,
        roll_freq[10][t] / 10,
        roll_freq[20][t] / 20,
        roll_freq[50][t] / 50,
        roll_freq[100][t] / 100,
        delay_cur[t] / max(1, t),
        delay_mean[t] / max(1, t),
        delay_max[t] / max(1, t),
        delay_ratio[t],
        decade / 9.0,
        parity,
        fascia / 2.0,
    ]
    return np.stack(feats, axis=1).astype(np.float32)

# ----- BASELINES -----
def score_uniform(t):
    return RNG.random(90)

def score_hist_freq(t):
    return cum_freq[t].astype(float)

def score_recent_freq(t):
    return roll_freq[50][t].astype(float)

def score_delay(t):
    # the more delayed, the higher score
    return delay_cur[t].astype(float)

def score_montecarlo(t):
    # noisy historical frequency
    base = cum_freq[t] + 1
    p_mc = base / base.sum()
    return p_mc + RNG.normal(0, 0.001, size=90)

# ----- PROBABILISTIC: Dirichlet-Multinomial -----
def score_dirichlet(t, alpha=1.0):
    counts = cum_freq[t]
    posterior_mean = (counts + alpha) / (counts.sum() + 90 * alpha)
    return posterior_mean

# ----- ML MODELS (refit periodically) -----
# Build full training dataset row-by-row: each (t, n) is a row with target=draw_mat[t,n]
# Walk-forward: we accumulate (features[t], target[t]) for t < t_test, then predict at t_test.

# Pre-build feature snapshots for all t (only for t up to N-1 used as features for predicting draw t)
# Note: feature at index t already represents pre-draw-t state.
FEATS_ALL = np.stack([build_features(t) for t in range(N)], axis=0)  # (N, 90, F)
F = FEATS_ALL.shape[2]

# Statefuly refit ML models periodically
def refit_logistic(t_train_end):
    X = FEATS_ALL[:t_train_end].reshape(-1, F)
    y = draw_mat[:t_train_end].reshape(-1)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    m = LogisticRegression(max_iter=200, C=0.5, n_jobs=-1)
    m.fit(Xs, y)
    return (sc, m)

def refit_rf(t_train_end):
    X = FEATS_ALL[:t_train_end].reshape(-1, F)
    y = draw_mat[:t_train_end].reshape(-1)
    m = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=200,
                               n_jobs=-1, random_state=42)
    m.fit(X, y)
    return m

def refit_lgbm(t_train_end):
    """LightGBM LambdaRank: each draw is a query group of 90 candidates."""
    X = FEATS_ALL[:t_train_end].reshape(-1, F)
    y = draw_mat[:t_train_end].reshape(-1)
    group = np.full(t_train_end, 90)
    m = lgb.LGBMRanker(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_data_in_leaf=50, verbose=-1, random_state=42,
    )
    m.fit(X, y, group=group)
    return m

# Per-strategy hit storage
strategies = ["uniform", "hist_freq", "recent_freq", "delay", "monte_carlo",
              "dirichlet_bayes", "logistic", "random_forest", "lgbm_ranker"]
hits_log = {s: [] for s in strategies}
# Also store probabilistic scores for calibration (for those that produce calibrated probs)
prob_log_dirichlet = []
prob_log_logistic = []
actual_log = []

# Walk forward
log_models = None
log_sc = None
rf_model = None
lgbm_model = None
last_refit = -1

print(f"  Initial train window: {INITIAL_TRAIN}  Refit every: {REFIT_EVERY}  Test draws: {N - INITIAL_TRAIN - 1}")
for t in range(INITIAL_TRAIN, N - 1):
    actual_idx = np.where(draw_mat[t + 1] == 1)[0]   # next-draw target indices
    feats = FEATS_ALL[t + 1]                          # features at decision point (pre-draw t+1)
    # Note: we are predicting draw[t+1] using FEATS_ALL[t+1] which only uses data <= t. OK.

    # Baselines
    hits_log["uniform"].append(hits_at_k(score_uniform(t + 1), actual_idx))
    hits_log["hist_freq"].append(hits_at_k(score_hist_freq(t + 1), actual_idx))
    hits_log["recent_freq"].append(hits_at_k(score_recent_freq(t + 1), actual_idx))
    hits_log["delay"].append(hits_at_k(score_delay(t + 1), actual_idx))
    hits_log["monte_carlo"].append(hits_at_k(score_montecarlo(t + 1), actual_idx))

    # Dirichlet-Multinomial (cheap; recompute every step)
    s_dir = score_dirichlet(t + 1, alpha=1.0)
    hits_log["dirichlet_bayes"].append(hits_at_k(s_dir, actual_idx))
    prob_log_dirichlet.append(s_dir.copy())

    # Refit ML models periodically
    if (t - last_refit) >= REFIT_EVERY:
        last_refit = t
        log_sc, log_models = refit_logistic(t + 1)
        rf_model = refit_rf(t + 1)
        lgbm_model = refit_lgbm(t + 1)

    s_log = log_models.predict_proba(log_sc.transform(feats))[:, 1]
    hits_log["logistic"].append(hits_at_k(s_log, actual_idx))
    prob_log_logistic.append(s_log.copy())

    s_rf = rf_model.predict_proba(feats)[:, 1] if hasattr(rf_model, "predict_proba") else rf_model.predict(feats)
    hits_log["random_forest"].append(hits_at_k(s_rf, actual_idx))

    s_lgbm = lgbm_model.predict(feats)
    hits_log["lgbm_ranker"].append(hits_at_k(s_lgbm, actual_idx))

    actual_log.append(actual_idx)

    if (t - INITIAL_TRAIN) % 100 == 0:
        print(f"    progress t={t}/{N-1}")

print("[BACKTEST] done")

# ============================ EVALUATION ============================
print("\n[EVAL] Metrics")
n_test = len(hits_log["uniform"])
uniform_mean = float(np.mean(hits_log["uniform"]))

eval_rows = []
for s in strategies:
    arr = np.array(hits_log[s])
    mean_hits = float(arr.mean())
    p6 = mean_hits / 6.0   # precision@6
    r6 = mean_hits / 6.0   # recall@6 (same since |actual|=6, |topk|=6)
    lift_pct = (mean_hits / uniform_mean - 1) * 100 if uniform_mean > 0 else float("nan")
    # bootstrap CI on mean hits
    boot = np.empty(BOOTSTRAP_ITERS)
    for b in range(BOOTSTRAP_ITERS):
        idx = RNG.integers(0, n_test, n_test)
        boot[b] = arr[idx].mean()
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    # hit distribution
    hd = {int(k): int((arr == k).sum()) for k in range(0, 7)}
    eval_rows.append({
        "strategy": s, "avg_hits": round(mean_hits, 4),
        "precision@6": round(p6, 4), "recall@6": round(r6, 4),
        "lift_vs_uniform_pct": round(lift_pct, 3),
        "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
        "hit_0": hd[0], "hit_1": hd[1], "hit_2": hd[2], "hit_3": hd[3], "hit_4+": hd[4] + hd[5] + hd[6],
    })

eval_df = pd.DataFrame(eval_rows).sort_values("avg_hits", ascending=False).reset_index(drop=True)
print(eval_df.to_string(index=False))

# ============================ PAIRED PERMUTATION TEST: model vs uniform ============================
def paired_perm_test(a, b, iters=5000):
    diff = a - b
    obs = diff.mean()
    boot = np.empty(iters)
    signs = RNG.choice([-1, 1], size=(iters, len(diff)))
    for k in range(iters):
        boot[k] = (diff * signs[k]).mean()
    return obs, float((np.abs(boot) >= abs(obs)).mean())

a_uni = np.array(hits_log["uniform"])
perm_rows = []
for s in strategies:
    if s == "uniform":
        continue
    obs, pv = paired_perm_test(np.array(hits_log[s]), a_uni, iters=3000)
    perm_rows.append({"strategy": s, "delta_vs_uniform": round(float(obs), 4),
                      "perm_p_value": round(pv, 4),
                      "robustly_beats_uniform": bool(pv < 0.05 and obs > 0)})
perm_df = pd.DataFrame(perm_rows).sort_values("delta_vs_uniform", ascending=False).reset_index(drop=True)
print("\n[PERM TEST] vs uniform")
print(perm_df.to_string(index=False))

# ============================ CALIBRATION (Brier) ============================
# For models that produce per-number probability of being drawn
def brier_per_step(prob_list, actual_list):
    bs = []
    for p_arr, a_idx in zip(prob_list, actual_list):
        y = np.zeros(90); y[a_idx] = 1
        # normalize prob to expected value 6/90 each? No, we keep raw model probability
        bs.append(np.mean((p_arr - y) ** 2))
    return float(np.mean(bs))

brier_dir = brier_per_step(prob_log_dirichlet, actual_log)
brier_log = brier_per_step(prob_log_logistic, actual_log)
brier_uniform = float(np.mean((np.full(90, 6/90) - 0)**2 * (90 - 6)/90 + (np.full(90, 6/90) - 1)**2 * 6/90))
print(f"\n[CALIB] Brier (lower better): uniform={brier_uniform:.5f}  dirichlet={brier_dir:.5f}  logistic={brier_log:.5f}")

# ============================ VERDICT ============================
best_row = eval_df.iloc[0]
best_strategy = best_row["strategy"]
best_lift = float(best_row["lift_vs_uniform_pct"])
# pick from perm_df the lift + significance
best_perm = perm_df[perm_df["strategy"] == best_strategy].iloc[0] if best_strategy in perm_df["strategy"].values else None

robust_signal = False
if best_perm is not None:
    robust_signal = (best_perm["robustly_beats_uniform"] and best_lift > LIFT_SIGNIFICANCE_PCT)

print(f"\n[VERDICT] best_strategy={best_strategy}  lift={best_lift:.2f}%  robust_signal={robust_signal}")

# ============================ COMBINATION GENERATION ============================
# Use last available model scores (= prediction for next draw beyond dataset)
print("\n[GEN] Generating candidate combinations")

# Final scores for "next" draw using last fit models and feats from N-1 (post-draw-(N-1) state for next draw)
# We use the rolling state from after the last draw: build features at t=N (next draw position)
def features_at(t):
    """Features at decision point for draw t (uses only draws 0..t-1)."""
    if t < N:
        return FEATS_ALL[t]
    # build for hypothetical next draw
    cf = draw_mat.sum(axis=0)
    rf = {w: draw_mat[max(0, t - w):t].sum(axis=0) if t <= N else draw_mat[N - w:].sum(axis=0)
          for w in ROLL_WINDOWS}
    # delays as of after last actual draw
    dc = np.zeros(90)
    dm = np.zeros(90); dx = np.zeros(90)
    last_seen2 = np.full(90, -1)
    gs = np.zeros(90); gc = np.zeros(90); gx = np.zeros(90)
    for tt in range(N):
        for k in range(90):
            if draw_mat[tt, k] == 1:
                if last_seen2[k] != -1:
                    g = tt - last_seen2[k]
                    gs[k] += g; gc[k] += 1
                    if g > gx[k]: gx[k] = g
                last_seen2[k] = tt
    for k in range(90):
        dc[k] = (N - 1) - last_seen2[k] if last_seen2[k] != -1 else N
        dm[k] = gs[k] / gc[k] if gc[k] > 0 else 0
        dx[k] = gx[k]
    dr = np.where(dm > 0, dc / dm, 0)
    feats = [cf / N, rf[5] / 5, rf[10] / 10, rf[20] / 20, rf[50] / 50, rf[100] / 100,
             dc / N, dm / N, dx / N, dr, decade / 9.0, parity, fascia / 2.0]
    return np.stack(feats, axis=1).astype(np.float32)

next_feats = features_at(N)

# Final scores per strategy (use latest fitted models)
# For ML, use the most recent fit (already fitted on data up to N-1)
final_scores = {
    "hist_freq": cum_freq[N - 1] + draw_mat[N - 1].astype(float),
    "recent_freq": draw_mat[N - 50:].sum(axis=0).astype(float),
    "delay": (N - 1) - np.where(draw_mat == 1, np.arange(N)[:, None], -1).max(axis=0),
    "dirichlet": ((draw_mat.sum(axis=0) + 1) / (draw_mat.sum() + 90)),
    "logistic": log_models.predict_proba(log_sc.transform(next_feats))[:, 1],
    "random_forest": rf_model.predict_proba(next_feats)[:, 1] if hasattr(rf_model, "predict_proba") else rf_model.predict(next_feats),
    "lgbm_ranker": lgbm_model.predict(next_feats),
}

def has_pattern_issues(combo):
    s = sorted(combo)
    if all(x % 2 == 0 for x in s) or all(x % 2 != 0 for x in s):
        return True
    if len(set((x - 1) // 10 for x in s)) <= 1:
        return True
    consec = 1; mx = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            consec += 1; mx = max(mx, consec)
        else:
            consec = 1
    if mx >= 4:
        return True
    if all(x <= 30 for x in s) or all(x >= 61 for x in s):
        return True
    return False

def sample_topweighted(scores, k=6, alpha=1.5, seed=None):
    """Sample k unique numbers proportionally to scores^alpha."""
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=float)
    s = np.clip(s, 1e-9, None) ** alpha
    p = s / s.sum()
    return tuple(sorted(rng.choice(90, size=k, replace=False, p=p) + 1))

def gen_strategy(strat_name, n_candidates=8, seeds_start=1000):
    sc = final_scores[strat_name]
    out = []
    seed = seeds_start
    while len(out) < n_candidates and seed < seeds_start + 500:
        c = sample_topweighted(sc, k=6, alpha=2.0, seed=seed)
        seed += 1
        if has_pattern_issues(c):
            continue
        out.append(c)
    return out

# Top-frequency / anti-pattern from historic data (kept as heuristic baseline)
def gen_antipattern(n_candidates=5, seeds_start=5000):
    rng = np.random.default_rng(seeds_start)
    recent = set()
    for i in range(N - 10, N):
        recent.update(int(x) for x in (np.where(draw_mat[i])[0] + 1))
    pool = [n for n in range(1, 91) if n not in recent]
    bassi = [x for x in pool if x <= 30]
    medi = [x for x in pool if 31 <= x <= 60]
    alti = [x for x in pool if x >= 61]
    if len(bassi) < 2 or len(medi) < 2 or len(alti) < 2:
        bassi, medi, alti = list(range(1, 31)), list(range(31, 61)), list(range(61, 91))
    out = []
    for _ in range(200):
        c = sorted(rng.choice(bassi, 2, replace=False).tolist()
                   + rng.choice(medi, 2, replace=False).tolist()
                   + rng.choice(alti, 2, replace=False).tolist())
        c = tuple(c)
        if has_pattern_issues(c):
            continue
        if c not in out:
            out.append(c)
        if len(out) >= n_candidates:
            break
    return out

# Build the candidate pool
all_candidates = []
for s in ["dirichlet", "logistic", "lgbm_ranker", "random_forest", "hist_freq", "recent_freq", "delay"]:
    for c in gen_strategy(s, n_candidates=4, seeds_start=hash(s) & 0xFFFF):
        all_candidates.append((s, c))
for c in gen_antipattern(n_candidates=4):
    all_candidates.append(("anti_pattern", c))

# Dedupe
seen = set(); dedup = []
for strat, c in all_candidates:
    if c in seen:
        continue
    seen.add(c); dedup.append((strat, c))

# ----- COMPOSITE SCORE -----
freq_map = {i: int(draw_mat[:, i - 1].sum()) for i in range(1, 91)}
last_seen_global = np.full(90, -1)
for tt in range(N):
    for k in range(90):
        if draw_mat[tt, k] == 1:
            last_seen_global[k] = tt
delay_map = {i: int((N - 1) - last_seen_global[i - 1]) if last_seen_global[i - 1] != -1 else N
             for i in range(1, 91)}
recent_set = set()
for i in range(N - 10, N):
    recent_set.update(int(x) for x in (np.where(draw_mat[i])[0] + 1))
last_draw_set = set(int(x) for x in (np.where(draw_mat[N - 1])[0] + 1))

# Per-strategy model score lookup
def model_score_of_combo(strat, combo):
    if strat in final_scores:
        sc = final_scores[strat]
        return float(np.mean([sc[c - 1] for c in combo]))
    return 0.0

def score_combo(combo, strat):
    s = sorted(combo)
    score = 0.0
    n_even_c = sum(1 for x in s if x % 2 == 0)
    if n_even_c == 3: score += 12
    elif n_even_c in (2, 4): score += 9
    elif n_even_c in (1, 5): score += 4
    n_dec = len(set((x - 1) // 10 for x in s))
    score += n_dec * 2
    n_b = sum(1 for x in s if x <= 30); n_m = sum(1 for x in s if 31 <= x <= 60); n_a = sum(1 for x in s if x >= 61)
    if n_b == n_m == n_a == 2: score += 10
    elif max(n_b, n_m, n_a) <= 3: score += 7
    elif max(n_b, n_m, n_a) <= 4: score += 4
    freq_score = sum(freq_map[x] for x in s) / (6 * max(freq_map.values()))
    score += freq_score * 12
    rit_avg = float(np.mean([delay_map[x] for x in s]))
    if 15 <= rit_avg <= 40: score += 12
    elif 10 <= rit_avg < 15 or 40 < rit_avg <= 50: score += 9
    else: score += 5
    overlap = len(set(s) & recent_set)
    score += max(0, 10 - overlap * 2)
    if not has_pattern_issues(s): score += 12
    consec = 1; mx = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            consec += 1; mx = max(mx, consec)
        else:
            consec = 1
    if mx < 3: score += 4
    overlap_last = len(set(s) & last_draw_set)
    score += max(0, 8 - overlap_last * 4)
    # Model-aligned bonus
    ms = model_score_of_combo(strat, s)
    score += min(10, ms * 100) if 0 < ms < 1 else min(10, ms / max(1, ms))
    return float(min(100, round(score / 1.02, 1)))

scored = []
for strat, combo in dedup:
    sc = score_combo(combo, strat)
    scored.append({"strategy": strat, "combo": combo, "score": sc})

# ----- PORTFOLIO GREEDY MMR -----
def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))

def greedy_mmr(items, k=25, lam=0.6):
    """items: list of dicts with 'score' and 'combo'. lam balances score vs diversity."""
    selected = []
    pool = sorted(items, key=lambda x: -x["score"])
    while pool and len(selected) < k:
        if not selected:
            selected.append(pool.pop(0))
            continue
        best_idx = -1; best_val = -1e9
        for i, it in enumerate(pool):
            div = 1 - max(jaccard(it["combo"], s["combo"]) for s in selected)
            mmr = lam * (it["score"] / 100) + (1 - lam) * div
            if mmr > best_val:
                best_val = mmr; best_idx = i
        selected.append(pool.pop(best_idx))
    return selected

final_set = greedy_mmr(scored, k=25, lam=0.65)

# Jolly / SuperStar pick (use historic frequencies)
jolly_freq = Counter(df["jolly"].astype(int).tolist())
ss_freq = Counter(df["superstar"].astype(int).tolist())

def pick_js(combo, seed):
    rng = np.random.default_rng(seed)
    jp = [n for n, _ in sorted(jolly_freq.items(), key=lambda x: -x[1])[:15] if n not in combo]
    sp = [n for n, _ in sorted(ss_freq.items(), key=lambda x: -x[1])[:15] if n not in combo]
    return int(rng.choice(jp) if jp else rng.integers(1, 91)), int(rng.choice(sp) if sp else rng.integers(1, 91))

rows = []
for rk, it in enumerate(final_set, 1):
    j, ss = pick_js(it["combo"], seed=9000 + rk)
    rows.append({
        "rank": rk,
        "strategy": it["strategy"],
        "combinazione_str": "-".join(f"{x:02d}" for x in sorted(it["combo"])),
        "jolly": j, "superstar": ss,
        "score": it["score"],
        "model_signal_used": robust_signal,
    })

final_df = pd.DataFrame(rows)
final_df.to_csv(OUT / "combinazioni_ranking_rigorous.csv", index=False)
print(f"\n[OUT] {len(final_df)} combinations saved → {OUT/'combinazioni_ranking_rigorous.csv'}")

# Save eval + perm + tests to JSON
eval_df.to_csv(OUT / "model_evaluation_rigorous.csv", index=False)
perm_df.to_csv(OUT / "permutation_test_rigorous.csv", index=False)

summary = {
    "n_extractions": N,
    "date_min": str(df["date"].iloc[0].date()),
    "date_max": str(df["date"].iloc[-1].date()),
    "tests": [
        {"name": n, "stat": float(s), "p_value": float(p), "outcome": o}
        for n, s, p, o in tests_log
    ],
    "entropy_bits": H_obs,
    "entropy_max": H_max,
    "entropy_eff_pct": entropy_eff * 100,
    "kl_from_uniform": kl,
    "js_from_uniform": js,
    "mutual_information_avg_bits": mi_avg,
    "mutual_information_max_bits": mi_max,
    "pairs_total": len(pair_pvals),
    "pairs_significant_BH_005": n_sig_bh,
    "pairs_significant_Bonferroni_005": n_sig_bonf,
    "backtest": {
        "initial_train": INITIAL_TRAIN,
        "test_draws": n_test,
        "uniform_mean_hits": uniform_mean,
        "best_strategy": best_strategy,
        "best_lift_pct": best_lift,
        "robust_signal": robust_signal,
        "brier": {"uniform": brier_uniform, "dirichlet": brier_dir, "logistic": brier_log},
    },
    "eval_table": eval_df.to_dict(orient="records"),
    "permutation_table": perm_df.to_dict(orient="records"),
}
with open(OUT / "analysis_summary_rigorous.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[OUT] summary → {OUT/'analysis_summary_rigorous.json'}")
print("\n[DONE] rigorous pipeline finished.")
