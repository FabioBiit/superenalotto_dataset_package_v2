"""
Phase C — Generazione di 20-25 combinazioni candidate
======================================================
Premessa scientifica (validata in Fase B):
  G0 (uniforme senza reimmissione) è il miglior modello generativo per il dato.
  Nessun modello mostra lift predittivo robusto sopra l'uniforme.
  Le combinazioni qui generate sono quindi candidate EURISTICHE,
  ottimizzate per qualità strutturale (parità, decadi, balance, anti-pattern)
  e per diversità di portafoglio (MMR greedy).
  NON sono previsioni.

Strategie:
  1. anti_pattern_balanced — 2 bassi + 2 medi + 2 alti, parità 3:3, ≥5 decadi
  2. anti_recent — nessun numero dalle ultime 10 estrazioni
  3. delay_weighted — pesato sui numeri con maggior ritardo corrente
  4. freq_weighted — pesato sulla frequenza storica (≈ uniforme)
  5. mixed_balance — bilanciato su tutti gli assi
  6. monte_carlo_uniform — uniforme + filtro qualità
  7. portfolio_diverse — massima distanza Jaccard dai candidati già selezionati

Output:
  inference/phaseC_combinations.csv      — ranking finale Top-25
  inference/phaseC_summary.json          — parametri + stats + warning
"""

from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------- CONFIG ----------------
BASE = Path(r"C:\Users\kyros\OneDrive\Desktop\superenalotto_dataset_package_v2\superenalotto_dataset_package_v2")
CSV = BASE / "data" / "superenalotto_5.5_anni_extraction.csv"
OUT = BASE / "inference"
OUT.mkdir(exist_ok=True)

N_TARGET = 25            # final portfolio size
N_CANDIDATES_PER_STRAT = 8
MMR_LAMBDA = 0.65        # balance score vs diversity

WARNING_TEXT = (
    "Le combinazioni generate non sono previsioni. Sono il risultato di criteri "
    "statistici, euristici, machine learning e teoria delle decisioni applicati "
    "su dati storici. Nel SuperEnalotto ogni combinazione ha esattamente la stessa "
    "probabilità teorica di essere estratta se il processo è uniforme e indipendente."
)

# ---------------- LOAD ----------------
COLS = ["date", "contest_number", "n1", "n2", "n3", "n4", "n5", "n6", "jolly", "superstar"]
df = pd.read_csv(CSV, header=0, names=COLS)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "contest_number"]).reset_index(drop=True)
MAIN = ["n1", "n2", "n3", "n4", "n5", "n6"]
N = len(df)

draw_mat = np.zeros((N, 90), dtype=np.int8)
for i in range(N):
    for c in MAIN:
        draw_mat[i, int(df.iloc[i][c]) - 1] = 1

# ---------------- STATISTICHE STORICHE ----------------
freq_arr = draw_mat.sum(axis=0).astype(int)        # (90,)
freq_map = {i + 1: int(freq_arr[i]) for i in range(90)}

# Ritardi correnti
last_seen = np.full(90, -1, dtype=int)
for t in range(N):
    for k in range(90):
        if draw_mat[t, k] == 1:
            last_seen[k] = t
delay_arr = np.array([(N - 1) - last_seen[k] if last_seen[k] != -1 else N for k in range(90)])
delay_map = {i + 1: int(delay_arr[i]) for i in range(90)}

# Numeri delle ultime 10 estrazioni
recent10 = set()
for i in range(N - 10, N):
    recent10.update(int(x) for x in (np.where(draw_mat[i])[0] + 1))

# Numeri dell'ultima estrazione
last_draw = set(int(x) for x in (np.where(draw_mat[N - 1])[0] + 1))

# Frequenze Jolly e SuperStar
jolly_freq = Counter(df["jolly"].astype(int).tolist())
ss_freq = Counter(df["superstar"].astype(int).tolist())

print(f"[LOAD] N={N} draws, last_draw={sorted(last_draw)}")

# ---------------- VINCOLI STRUTTURALI ----------------
def has_pattern_issues(combo):
    s = sorted(combo)
    if all(x % 2 == 0 for x in s) or all(x % 2 != 0 for x in s):
        return True  # tutti pari o tutti dispari
    if len(set((x - 1) // 10 for x in s)) <= 1:
        return True  # tutti in una decade
    consec = 1
    mx = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            consec += 1
            mx = max(mx, consec)
        else:
            consec = 1
    if mx >= 4:
        return True  # 4+ consecutivi
    if all(x <= 30 for x in s) or all(x >= 61 for x in s):
        return True  # tutti bassi o tutti alti
    return False

def quality_metrics(combo):
    s = sorted(combo)
    n_even = sum(1 for x in s if x % 2 == 0)
    n_dec = len(set((x - 1) // 10 for x in s))
    n_b = sum(1 for x in s if x <= 30)
    n_m = sum(1 for x in s if 31 <= x <= 60)
    n_a = sum(1 for x in s if x >= 61)
    return {"n_even": n_even, "n_dec": n_dec, "n_b": n_b, "n_m": n_m, "n_a": n_a}

# ---------------- STRATEGIE DI GENERAZIONE ----------------
def gen_anti_pattern_balanced(n_out, seed_base):
    out, seed = [], seed_base
    while len(out) < n_out and seed < seed_base + 2000:
        rng = np.random.default_rng(seed)
        seed += 1
        bassi = rng.choice(np.arange(1, 31), size=2, replace=False)
        medi = rng.choice(np.arange(31, 61), size=2, replace=False)
        alti = rng.choice(np.arange(61, 91), size=2, replace=False)
        c = tuple(sorted(int(x) for x in np.concatenate([bassi, medi, alti])))
        if has_pattern_issues(c):
            continue
        m = quality_metrics(c)
        if m["n_even"] not in (2, 3, 4):
            continue
        if m["n_dec"] < 5:
            continue
        if c not in out:
            out.append(c)
    return out

def gen_anti_recent(n_out, seed_base):
    pool = [x for x in range(1, 91) if x not in recent10]
    if len(pool) < 30:
        pool = list(range(1, 91))
    out, seed = [], seed_base
    while len(out) < n_out and seed < seed_base + 2000:
        rng = np.random.default_rng(seed)
        seed += 1
        bassi = [x for x in pool if x <= 30]
        medi = [x for x in pool if 31 <= x <= 60]
        alti = [x for x in pool if x >= 61]
        if len(bassi) < 2 or len(medi) < 2 or len(alti) < 2:
            continue
        c = tuple(sorted(
            rng.choice(bassi, size=2, replace=False).tolist()
            + rng.choice(medi, size=2, replace=False).tolist()
            + rng.choice(alti, size=2, replace=False).tolist()
        ))
        if has_pattern_issues(c):
            continue
        m = quality_metrics(c)
        if m["n_dec"] < 5 or m["n_even"] not in (2, 3, 4):
            continue
        if c not in out:
            out.append(c)
    return out

def gen_delay_weighted(n_out, seed_base):
    """Pesato sui ritardatari (euristica, non predittiva)."""
    weights = (delay_arr + 1).astype(float)
    p = weights / weights.sum()
    out, seed = [], seed_base
    while len(out) < n_out and seed < seed_base + 2000:
        rng = np.random.default_rng(seed)
        seed += 1
        c = tuple(sorted(rng.choice(np.arange(1, 91), size=6, replace=False, p=p).tolist()))
        if has_pattern_issues(c):
            continue
        m = quality_metrics(c)
        if m["n_dec"] < 4 or m["n_even"] not in (2, 3, 4):
            continue
        if c not in out:
            out.append(c)
    return out

def gen_freq_weighted(n_out, seed_base):
    """Pesato sulla frequenza storica (≈ uniforme dato G0)."""
    weights = (freq_arr + 1).astype(float)
    p = weights / weights.sum()
    out, seed = [], seed_base
    while len(out) < n_out and seed < seed_base + 2000:
        rng = np.random.default_rng(seed)
        seed += 1
        c = tuple(sorted(rng.choice(np.arange(1, 91), size=6, replace=False, p=p).tolist()))
        if has_pattern_issues(c):
            continue
        m = quality_metrics(c)
        if m["n_dec"] < 4 or m["n_even"] not in (2, 3, 4):
            continue
        if c not in out:
            out.append(c)
    return out

def gen_mixed_balance(n_out, seed_base):
    """Bilanciata: media tra freq e delay weighted, con vincoli di qualità."""
    w_freq = (freq_arr + 1).astype(float)
    w_delay = (delay_arr + 1).astype(float)
    w = 0.5 * w_freq / w_freq.sum() + 0.5 * w_delay / w_delay.sum()
    out, seed = [], seed_base
    while len(out) < n_out and seed < seed_base + 2000:
        rng = np.random.default_rng(seed)
        seed += 1
        c = tuple(sorted(rng.choice(np.arange(1, 91), size=6, replace=False, p=w).tolist()))
        if has_pattern_issues(c):
            continue
        m = quality_metrics(c)
        if m["n_dec"] < 5 or m["n_even"] not in (2, 3, 4):
            continue
        if c not in out:
            out.append(c)
    return out

def gen_monte_carlo_uniform(n_out, seed_base):
    """G0 puro + filtro qualità."""
    out, seed = [], seed_base
    while len(out) < n_out and seed < seed_base + 2000:
        rng = np.random.default_rng(seed)
        seed += 1
        c = tuple(sorted(rng.choice(np.arange(1, 91), size=6, replace=False).tolist()))
        if has_pattern_issues(c):
            continue
        m = quality_metrics(c)
        if m["n_dec"] < 5 or m["n_even"] not in (2, 3, 4):
            continue
        if c not in out:
            out.append(c)
    return out

# ---------------- POOL DI CANDIDATI ----------------
print("[GEN] Generating candidate combinations...")
candidates = []
strategies = [
    ("anti_pattern_balanced", gen_anti_pattern_balanced, 1000),
    ("anti_recent",           gen_anti_recent,           2000),
    ("delay_weighted",        gen_delay_weighted,        3000),
    ("freq_weighted",         gen_freq_weighted,         4000),
    ("mixed_balance",         gen_mixed_balance,         5000),
    ("monte_carlo_uniform",   gen_monte_carlo_uniform,   6000),
]
for name, fn, seed_base in strategies:
    combos = fn(N_CANDIDATES_PER_STRAT, seed_base)
    print(f"  {name}: {len(combos)} candidates")
    for c in combos:
        candidates.append({"strategy": name, "combo": c})

# Deduplica mantenendo prima strategia che ha generato la combinazione
seen, dedup = set(), []
for it in candidates:
    if it["combo"] not in seen:
        seen.add(it["combo"])
        dedup.append(it)
print(f"[GEN] {len(candidates)} → {len(dedup)} unique")

# ---------------- SCORING ----------------
max_freq = max(freq_map.values())

def score_combo(combo):
    s = sorted(combo)
    score = 0.0
    m = quality_metrics(s)
    # Parity
    if m["n_even"] == 3: score += 12
    elif m["n_even"] in (2, 4): score += 9
    elif m["n_even"] in (1, 5): score += 4
    # Decades
    score += m["n_dec"] * 2
    # Range balance
    if m["n_b"] == m["n_m"] == m["n_a"] == 2:
        score += 10
    elif max(m["n_b"], m["n_m"], m["n_a"]) <= 3:
        score += 7
    elif max(m["n_b"], m["n_m"], m["n_a"]) <= 4:
        score += 4
    # Average frequency (normalized)
    freq_score = sum(freq_map[x] for x in s) / (6 * max_freq)
    score += freq_score * 12
    # Average delay band (sweet spot 15-40)
    rit_avg = float(np.mean([delay_map[x] for x in s]))
    if 15 <= rit_avg <= 40:
        score += 12
    elif 10 <= rit_avg < 15 or 40 < rit_avg <= 50:
        score += 9
    else:
        score += 5
    # Overlap with last 10 draws (prefer less)
    overlap_recent = len(set(s) & recent10)
    score += max(0, 10 - overlap_recent * 2)
    # No pattern issues
    if not has_pattern_issues(s):
        score += 12
    # Max consecutive < 3
    consec, mx = 1, 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            consec += 1
            mx = max(mx, consec)
        else:
            consec = 1
    if mx < 3:
        score += 4
    # Overlap with last draw (penalize)
    overlap_last = len(set(s) & last_draw)
    score += max(0, 8 - overlap_last * 4)
    # Range span (penalize tiny ranges)
    span = max(s) - min(s)
    if span >= 50:
        score += 6
    elif span >= 35:
        score += 3
    return float(min(100, round(score / 1.02, 1)))

for it in dedup:
    it["score"] = score_combo(it["combo"])

# ---------------- PORTFOLIO MMR ----------------
def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))

def greedy_mmr(items, k, lam=MMR_LAMBDA):
    pool = sorted(items, key=lambda x: -x["score"])
    selected = []
    while pool and len(selected) < k:
        if not selected:
            selected.append(pool.pop(0))
            continue
        best_i, best_val = -1, -1e9
        for i, it in enumerate(pool):
            div = 1 - max(jaccard(it["combo"], s["combo"]) for s in selected)
            mmr = lam * (it["score"] / 100) + (1 - lam) * div
            if mmr > best_val:
                best_val, best_i = mmr, i
        selected.append(pool.pop(best_i))
    return selected

final = greedy_mmr(dedup, k=N_TARGET, lam=MMR_LAMBDA)
print(f"[MMR] selected top {len(final)} via greedy MMR (λ={MMR_LAMBDA})")

# ---------------- JOLLY & SUPERSTAR ----------------
def pick_js(combo, seed):
    rng = np.random.default_rng(seed)
    jolly_pool = [n for n, _ in sorted(jolly_freq.items(), key=lambda x: -x[1])[:15] if n not in combo]
    ss_pool = [n for n, _ in sorted(ss_freq.items(), key=lambda x: -x[1])[:15] if n not in combo]
    j = int(rng.choice(jolly_pool)) if jolly_pool else int(rng.integers(1, 91))
    s = int(rng.choice(ss_pool)) if ss_pool else int(rng.integers(1, 91))
    return j, s

# ---------------- BUILD FINAL TABLE ----------------
rows = []
for rk, it in enumerate(final, 1):
    j, ss = pick_js(it["combo"], seed=9000 + rk)
    m = quality_metrics(it["combo"])
    rows.append({
        "rank": rk,
        "strategy": it["strategy"],
        "combinazione": "-".join(f"{x:02d}" for x in sorted(it["combo"])),
        "jolly": j,
        "superstar": ss,
        "score": it["score"],
        "n_even": m["n_even"],
        "n_dec": m["n_dec"],
        "rit_avg": round(float(np.mean([delay_map[x] for x in it["combo"]])), 2),
        "freq_avg": round(float(np.mean([freq_map[x] for x in it["combo"]])), 2),
        "overlap_last10": len(set(it["combo"]) & recent10),
        "warning": "NON_PREVISIONE",
    })

final_df = pd.DataFrame(rows)

# ---------------- EXPORT ----------------
out_csv = OUT / "phaseC_combinations.csv"
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    f.write(f"# {WARNING_TEXT}\n")
    final_df.to_csv(f, index=False)
print(f"[OUT] {out_csv}")

summary = {
    "warning": WARNING_TEXT,
    "method": "Phase C heuristic combination generation",
    "data_basis": {
        "n_extractions": int(N),
        "period": f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}",
        "phaseB_verdict": "G0 uniforme è il miglior modello; nessun signal predittivo robusto.",
    },
    "generation": {
        "strategies": [s[0] for s in strategies],
        "candidates_per_strategy": N_CANDIDATES_PER_STRAT,
        "total_candidates": len(candidates),
        "unique_candidates": len(dedup),
        "selected": len(final),
        "mmr_lambda": MMR_LAMBDA,
    },
    "constraints_applied": {
        "no_all_even_or_all_odd": True,
        "no_single_decade": True,
        "no_4plus_consecutive": True,
        "no_all_low_or_all_high": True,
        "min_decades": 5,
        "n_even_in_2_3_4": True,
    },
    "scoring_components": [
        "parity", "decades", "range_balance", "freq", "delay_band",
        "overlap_recent10", "no_pattern_issues", "max_consec",
        "overlap_last_draw", "range_span",
    ],
    "stats_per_combo": final_df.to_dict(orient="records"),
}
with open(OUT / "phaseC_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(f"[OUT] {OUT / 'phaseC_summary.json'}")
print()
print(f"=" * 80)
print(WARNING_TEXT)
print(f"=" * 80)
print()
print(final_df[["rank", "strategy", "combinazione", "jolly", "superstar", "score",
                "n_even", "n_dec", "rit_avg"]].to_string(index=False))
print()
print(f"=" * 80)
print(WARNING_TEXT)
print(f"=" * 80)
