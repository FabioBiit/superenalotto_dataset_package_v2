"""
SuperEnalotto - Rigorous statistical analysis
PhD-level approach: descriptive only, never predictive.
"""

import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import math
import random
import json
import os

random.seed(42)
np.random.seed(42)

CSV_PATH = r"C:\Users\kyros\OneDrive\Desktop\superenalotto_dataset_package_v2\superenalotto_dataset_package_v2\data\superenalotto_2025_2026.csv"

# --- 1) LOAD & SCHEMA RECONCILE ---------------------------------------------
# Declared header has 8 cols, schema-file says 13 cols, data has 10 cols.
# We reconcile by reading WITHOUT header and assigning explicit names.
COLUMNS = ["date", "contest_number", "n1", "n2", "n3", "n4", "n5", "n6", "jolly", "superstar"]
df = pd.read_csv(CSV_PATH, header=0, names=COLUMNS)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

# --- 2) DATA QUALITY CHECK --------------------------------------------------
dq_results = []

null_cnt = df.isnull().sum().sum()
dq_results.append(("Valori nulli", "OK" if null_cnt == 0 else "FAIL", f"{null_cnt} valori nulli totali", "Bassa" if null_cnt == 0 else "Alta"))

dup_rows = df.duplicated().sum()
dq_results.append(("Righe duplicate identiche", "OK" if dup_rows == 0 else "FAIL", f"{dup_rows} righe", "Bassa" if dup_rows == 0 else "Alta"))

# Duplicati di sestina (n1..n6 set uguale, indipendente da ordine)
df["sestina"] = df[MAIN_COLS].apply(lambda r: tuple(sorted(r.tolist())), axis=1)
dup_sestine = df["sestina"].duplicated().sum()
dq_results.append(("Sestine duplicate (set)", "OK" if dup_sestine == 0 else "WARN", f"{dup_sestine} sestine duplicate come set", "Bassa" if dup_sestine == 0 else "Media"))

# Range check 1..90 per numeri principali, jolly, superstar
def out_of_range(s, lo=1, hi=90):
    return ((s < lo) | (s > hi)).sum()

oor_main = sum(out_of_range(df[c]) for c in MAIN_COLS)
dq_results.append(("Range numeri principali 1..90", "OK" if oor_main == 0 else "FAIL", f"{oor_main} valori fuori range", "Bassa" if oor_main == 0 else "Alta"))

oor_jolly = out_of_range(df["jolly"])
dq_results.append(("Range Jolly 1..90", "OK" if oor_jolly == 0 else "FAIL", f"{oor_jolly} valori fuori range", "Bassa" if oor_jolly == 0 else "Alta"))

oor_ss = out_of_range(df["superstar"])
dq_results.append(("Range SuperStar 1..90", "OK" if oor_ss == 0 else "FAIL", f"{oor_ss} valori fuori range", "Bassa" if oor_ss == 0 else "Alta"))

# Sestine con duplicati interni
def has_internal_dup(row):
    nums = [row[c] for c in MAIN_COLS]
    return len(nums) != len(set(nums))

internal_dup = df.apply(has_internal_dup, axis=1).sum()
dq_results.append(("Sestine con duplicati interni", "OK" if internal_dup == 0 else "FAIL", f"{internal_dup} sestine non valide", "Bassa" if internal_dup == 0 else "Critica"))

# Numero di estrazioni
n_extractions = len(df)
date_min = df["date"].min().strftime("%Y-%m-%d")
date_max = df["date"].max().strftime("%Y-%m-%d")
dq_results.append(("N. estrazioni", "INFO", f"{n_extractions} estrazioni dal {date_min} al {date_max}", "Info"))

# contest_number reset per anno (verifica monotonia all'interno di ciascun anno)
df["year"] = df["date"].dt.year
contest_anomalies = 0
for yr, grp in df.groupby("year"):
    contests = grp["contest_number"].values
    if not all(contests[i] < contests[i+1] for i in range(len(contests)-1)):
        contest_anomalies += 1
dq_results.append(("Monotonia contest_number per anno", "OK" if contest_anomalies == 0 else "WARN", f"{contest_anomalies} anni con contest non monotono", "Bassa" if contest_anomalies == 0 else "Media"))

# Mancata presenza colonne winners_6, prize_6
dq_results.append(("Colonne winners_6 / prize_6 / source_url", "WARN", "Assenti nel CSV nonostante presenti nello schema dichiarato", "Media"))

# Date duplicate
dup_dates = df["date"].duplicated().sum()
dq_results.append(("Date duplicate", "OK" if dup_dates == 0 else "WARN", f"{dup_dates} date duplicate", "Bassa" if dup_dates == 0 else "Media"))

print("=== DATA QUALITY CHECK ===")
for r in dq_results:
    print(r)

# --- 3) FREQUENZE -----------------------------------------------------------
all_nums = df[MAIN_COLS].values.flatten()
freq = Counter(all_nums.tolist())
total_draws_per_num = sum(freq.values())  # = 6 * n_extractions

freq_df = pd.DataFrame({
    "numero": list(range(1, 91)),
    "frequenza": [freq.get(i, 0) for i in range(1, 91)]
})
freq_df["pct"] = freq_df["frequenza"] / total_draws_per_num * 100
freq_df["pct_atteso"] = 100/90
freq_df = freq_df.sort_values("frequenza", ascending=False).reset_index(drop=True)

top10_freq = freq_df.head(10)
bottom10_freq = freq_df.tail(10).iloc[::-1]

# Pari / dispari
n_even = sum(1 for x in all_nums if x % 2 == 0)
n_odd = sum(1 for x in all_nums if x % 2 != 0)

# Distribuzione per decine 1-9,10-19,...,80-89,90
def decile(x):
    if x == 90: return "90"
    return f"{(x//10)*10}-{(x//10)*10+9}"

decile_counts = Counter(decile(x) for x in all_nums)

# Bassa (1-30), media (31-60), alta (61-90)
def fascia(x):
    if x <= 30: return "Bassa (1-30)"
    if x <= 60: return "Media (31-60)"
    return "Alta (61-90)"
fascia_counts = Counter(fascia(x) for x in all_nums)

# Jolly
jolly_freq = Counter(df["jolly"].tolist())
jolly_top = sorted(jolly_freq.items(), key=lambda x: -x[1])[:10]

# SuperStar
ss_freq = Counter(df["superstar"].tolist())
ss_top = sorted(ss_freq.items(), key=lambda x: -x[1])[:10]

print("\n=== TOP 10 FREQUENTI ===")
print(top10_freq.to_string(index=False))
print("\n=== BOTTOM 10 (meno frequenti) ===")
print(bottom10_freq.to_string(index=False))
print(f"\nPari: {n_even} ({n_even/(n_even+n_odd)*100:.2f}%)")
print(f"Dispari: {n_odd} ({n_odd/(n_even+n_odd)*100:.2f}%)")
print(f"\nDecine: {dict(sorted(decile_counts.items()))}")
print(f"Fasce: {dict(fascia_counts)}")

# --- 4) RITARDI -------------------------------------------------------------
# Ritardo = estrazioni trascorse dall'ultima uscita
df_sorted = df.sort_values(["date", "contest_number"]).reset_index(drop=True)
N = len(df_sorted)

last_seen = {i: None for i in range(1, 91)}  # index dell'ultima estrazione
ritardi_correnti = {}
ritardo_max_storico = {i: 0 for i in range(1, 91)}
ritardo_medio = {}
ritardi_storici_per_num = {i: [] for i in range(1, 91)}

for idx, row in df_sorted.iterrows():
    nums = set([row[c] for c in MAIN_COLS])
    for num in range(1, 91):
        if num in nums:
            if last_seen[num] is not None:
                gap = idx - last_seen[num]  # ritardo accumulato prima di uscire
                ritardi_storici_per_num[num].append(gap)
                if gap > ritardo_max_storico[num]:
                    ritardo_max_storico[num] = gap
            last_seen[num] = idx

# Ritardo corrente alla fine
for num in range(1, 91):
    if last_seen[num] is None:
        ritardi_correnti[num] = N
    else:
        ritardi_correnti[num] = N - 1 - last_seen[num]
    if ritardi_storici_per_num[num]:
        ritardo_medio[num] = float(np.mean(ritardi_storici_per_num[num]))
    else:
        ritardo_medio[num] = float("nan")

rit_df = pd.DataFrame({
    "numero": list(range(1, 91)),
    "ritardo_attuale": [ritardi_correnti[i] for i in range(1, 91)],
    "ritardo_medio": [ritardo_medio[i] for i in range(1, 91)],
    "ritardo_max_storico": [ritardo_max_storico[i] for i in range(1, 91)],
    "freq": [freq.get(i, 0) for i in range(1, 91)],
})
rit_df["rapporto_rit"] = rit_df["ritardo_attuale"] / rit_df["ritardo_medio"].replace(0, np.nan)
rit_df = rit_df.sort_values("ritardo_attuale", ascending=False).reset_index(drop=True)

top_ritardatari = rit_df.head(15).copy()
hot_numbers = rit_df.sort_values("ritardo_attuale", ascending=True).head(15).copy()

print("\n=== TOP 15 RITARDATARI (ritardo corrente alto) ===")
print(top_ritardatari.to_string(index=False))
print("\n=== TOP 15 HOT NUMBERS (ritardo corrente basso) ===")
print(hot_numbers.to_string(index=False))

# --- 5) TEMPORALE -----------------------------------------------------------
df_sorted["month"] = df_sorted["date"].dt.to_period("M").astype(str)
recent_window_3m = df_sorted.tail(int(len(df_sorted)*0.20))  # ultime ~20% estrazioni come proxy "recenti"
# Calcolo migliore: ultime 36 estrazioni (~3 mesi a 12/mese) e ultime 72 (~6 mesi)
last_36 = df_sorted.tail(36)
last_72 = df_sorted.tail(72)

def freq_subset(sub):
    nums = sub[MAIN_COLS].values.flatten()
    c = Counter(nums.tolist())
    s = sum(c.values())
    return {k: v/s*100 for k, v in c.items()}

freq_total_pct = {i: freq.get(i,0)/total_draws_per_num*100 for i in range(1,91)}
freq_36_pct = freq_subset(last_36)
freq_72_pct = freq_subset(last_72)

# Top "scaldati" rispetto al totale
scaldati = sorted(((i, freq_36_pct.get(i,0) - freq_total_pct[i]) for i in range(1,91)), key=lambda x: -x[1])[:10]
raffreddati = sorted(((i, freq_36_pct.get(i,0) - freq_total_pct[i]) for i in range(1,91)), key=lambda x: x[1])[:10]

print(f"\nUltime 36 estrazioni - Top numeri 'scaldati': {scaldati}")
print(f"Ultime 36 estrazioni - Top numeri 'raffreddati': {raffreddati}")

# --- 6) COPPIE E TRIPLE -----------------------------------------------------
pair_counter = Counter()
triple_counter = Counter()
for _, row in df_sorted.iterrows():
    nums = sorted([row[c] for c in MAIN_COLS])
    for pair in combinations(nums, 2):
        pair_counter[pair] += 1
    for triple in combinations(nums, 3):
        triple_counter[triple] += 1

top_pairs = pair_counter.most_common(15)
top_triples = triple_counter.most_common(15)

# Coppie attese in casuale uniforme
n_pair_expected = N * 15 / (90*89/2)  # ogni estrazione ha C(6,2)=15 coppie su C(90,2) possibili
n_triple_expected = N * 20 / (90*89*88/6)

print(f"\nFrequenza media attesa per coppia in casuale uniforme: {n_pair_expected:.3f}")
print(f"Frequenza media attesa per tripla in casuale uniforme: {n_triple_expected:.3f}")
print("\nTop coppie ricorrenti:", top_pairs[:10])
print("Top triple ricorrenti:", top_triples[:10])

# --- 7) TEST DI CASUALITA ---------------------------------------------------
# Chi-quadro semplice (manuale, no scipy)
observed = np.array([freq.get(i, 0) for i in range(1, 91)], dtype=float)
expected = np.full(90, total_draws_per_num / 90)
chi2_stat = float(np.sum((observed - expected)**2 / expected))
df_chi2 = 89
# Valore critico chi2 89 df, alpha=0.05 ~ 112.022; alpha=0.01 ~ 122.94
chi2_critical_05 = 112.022
chi2_critical_01 = 122.942

# Entropia di Shannon (in bit), entropia massima = log2(90)
p = observed / observed.sum()
p_nz = p[p > 0]
entropy_bits = float(-np.sum(p_nz * np.log2(p_nz)))
entropy_max = math.log2(90)

# Monte Carlo: simulo M estrazioni "casuali" e calcolo chi2 distribuzione
M = 5000
mc_chi2 = []
rng = np.random.default_rng(123)
for _ in range(M):
    sim = []
    for _ in range(N):
        sim.extend(rng.choice(90, size=6, replace=False) + 1)
    obs_sim = np.bincount(sim, minlength=91)[1:91]
    expected_sim = np.full(90, sim.__len__()/90)
    mc_chi2.append(float(np.sum((obs_sim - expected_sim)**2 / expected_sim)))
mc_chi2 = np.array(mc_chi2)
p_value_empirical = float((mc_chi2 >= chi2_stat).mean())

print(f"\nChi-quadro osservato: {chi2_stat:.3f}")
print(f"Critical 5% (89 df): {chi2_critical_05}")
print(f"Critical 1% (89 df): {chi2_critical_01}")
print(f"Entropia Shannon: {entropy_bits:.4f} bit (max teorico {entropy_max:.4f})")
print(f"Efficienza entropica: {entropy_bits/entropy_max*100:.2f}%")
print(f"P-value empirico Monte Carlo (chi2 >= osservato): {p_value_empirical:.4f}")
print(f"MC chi2: mean={mc_chi2.mean():.2f}, std={mc_chi2.std():.2f}, p95={np.percentile(mc_chi2, 95):.2f}")

# --- 8) STRATEGIE DI GENERAZIONE --------------------------------------------
freq_map = {i: freq.get(i, 0) for i in range(1, 91)}
rit_map = {i: ritardi_correnti[i] for i in range(1, 91)}
rit_medio_map = {i: ritardo_medio[i] for i in range(1, 91)}

def is_valid_combo(combo):
    return len(set(combo)) == 6 and all(1 <= x <= 90 for x in combo)

def has_pattern_issues(combo):
    s = sorted(combo)
    # tutti pari o tutti dispari
    if all(x % 2 == 0 for x in s) or all(x % 2 != 0 for x in s):
        return True
    # tutti nella stessa decina
    deciles = set((x-1)//10 for x in s)
    if len(deciles) <= 1:
        return True
    # 4+ consecutivi
    consec = 1
    max_consec = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    if max_consec >= 4:
        return True
    # tutti bassi o tutti alti
    if all(x <= 30 for x in s) or all(x >= 61 for x in s):
        return True
    return False

# Strategia 1: Frequenza storica (top freq + qualche media)
def strat_frequenza(seed):
    rng = random.Random(seed)
    top_pool = [n for n, f in sorted(freq_map.items(), key=lambda x: -x[1])[:30]]
    mid_pool = [n for n, f in sorted(freq_map.items(), key=lambda x: -x[1])[25:55]]
    for _ in range(500):
        c = rng.sample(top_pool, 4) + rng.sample([x for x in mid_pool if x not in top_pool[:4]], 2)
        c = sorted(set(c))
        if len(c) == 6 and not has_pattern_issues(c):
            return c
    return sorted(rng.sample(top_pool, 6))

# Strategia 2: Ritardatari forti
def strat_ritardatari(seed):
    rng = random.Random(seed)
    pool = [n for n, _ in sorted(rit_map.items(), key=lambda x: -x[1])[:30]]
    for _ in range(500):
        c = sorted(rng.sample(pool, 6))
        if not has_pattern_issues(c):
            return c
    return sorted(rng.sample(pool, 6))

# Strategia 3: Bilanciata
def strat_bilanciata(seed):
    rng = random.Random(seed)
    top_freq = [n for n, _ in sorted(freq_map.items(), key=lambda x: -x[1])[:25]]
    top_rit = [n for n, _ in sorted(rit_map.items(), key=lambda x: -x[1])[:25]]
    mid = [n for n in range(1, 91) if n not in top_freq[:10] and n not in top_rit[:10]]
    for _ in range(500):
        c = rng.sample(top_freq, 2) + rng.sample(top_rit, 2) + rng.sample(mid, 2)
        c = sorted(set(c))
        if len(c) == 6 and not has_pattern_issues(c):
            # controlla bilanciamento pari/dispari (2-4 pari)
            n_even_c = sum(1 for x in c if x % 2 == 0)
            if 2 <= n_even_c <= 4:
                return c
    return sorted(c) if len(c)==6 else sorted(rng.sample(range(1,91), 6))

# Strategia 4: Monte Carlo guided (campionamento pesato sulle frequenze)
def strat_montecarlo(seed):
    rng = np.random.default_rng(seed)
    weights = np.array([freq_map[i] for i in range(1, 91)], dtype=float)
    weights = weights / weights.sum()
    for _ in range(500):
        c = sorted(rng.choice(range(1, 91), size=6, replace=False, p=weights).tolist())
        if not has_pattern_issues(c):
            return c
    return sorted(rng.choice(range(1, 91), size=6, replace=False, p=weights).tolist())

# Strategia 5: Anti-pattern / Massima diversificazione
def strat_antipattern(seed):
    rng = random.Random(seed)
    # Estrazioni recenti
    recent_nums = set()
    for _, row in df_sorted.tail(10).iterrows():
        for c in MAIN_COLS:
            recent_nums.add(row[c])
    pool_diverso = [n for n in range(1, 91) if n not in recent_nums]
    for _ in range(500):
        # Forziamo distribuzione: 2 bassi 1-30, 2 medi 31-60, 2 alti 61-90
        bassi = [x for x in pool_diverso if x <= 30]
        medi = [x for x in pool_diverso if 31 <= x <= 60]
        alti = [x for x in pool_diverso if x >= 61]
        if len(bassi) < 2 or len(medi) < 2 or len(alti) < 2:
            bassi = list(range(1, 31))
            medi = list(range(31, 61))
            alti = list(range(61, 91))
        c = sorted(rng.sample(bassi, 2) + rng.sample(medi, 2) + rng.sample(alti, 2))
        n_even_c = sum(1 for x in c if x % 2 == 0)
        if not has_pattern_issues(c) and 2 <= n_even_c <= 4:
            # Massima dispersione: tutte le decine distinte
            if len(set((x-1)//10 for x in c)) >= 5:
                return c
    return c

# Genera 6 combinazioni per ciascuna delle 5 strategie => 30 totali
combos = []
for i in range(6):
    combos.append(("Frequenza storica", strat_frequenza(1000+i)))
    combos.append(("Ritardatari forti", strat_ritardatari(2000+i)))
    combos.append(("Bilanciata", strat_bilanciata(3000+i)))
    combos.append(("Monte Carlo guided", strat_montecarlo(4000+i)))
    combos.append(("Anti-pattern", strat_antipattern(5000+i)))

# --- 9) JOLLY / SUPERSTAR pick (top freq, evitando coincidenze con i 6) ----
def pick_jolly_ss(combo, seed):
    rng = random.Random(seed)
    jolly_top_pool = [n for n,_ in sorted(jolly_freq.items(), key=lambda x: -x[1])[:15] if n not in combo]
    ss_top_pool = [n for n,_ in sorted(ss_freq.items(), key=lambda x: -x[1])[:15] if n not in combo]
    j = rng.choice(jolly_top_pool) if jolly_top_pool else rng.randint(1, 90)
    s = rng.choice(ss_top_pool) if ss_top_pool else rng.randint(1, 90)
    return j, s

# --- 10) SCORING EURISTICO --------------------------------------------------
recent_nums_set = set()
for _, row in df_sorted.tail(10).iterrows():
    for c in MAIN_COLS:
        recent_nums_set.add(row[c])

last_extraction_set = set([df_sorted.iloc[-1][c] for c in MAIN_COLS])

def score_combo(combo):
    s = sorted(combo)
    score = 0.0
    # 1) Equilibrio pari/dispari (max 12 pt)
    n_even_c = sum(1 for x in s if x % 2 == 0)
    if n_even_c == 3: score += 12
    elif n_even_c in (2, 4): score += 9
    elif n_even_c in (1, 5): score += 4
    else: score += 0
    # 2) Distribuzione per decina (max 12 pt)
    deciles_count = len(set((x-1)//10 for x in s))
    score += deciles_count * 2  # 6 distinte = 12
    # 3) Distribuzione fasce (max 10 pt)
    n_b = sum(1 for x in s if x <= 30)
    n_m = sum(1 for x in s if 31 <= x <= 60)
    n_a = sum(1 for x in s if x >= 61)
    if n_b == n_m == n_a == 2: score += 10
    elif max(n_b, n_m, n_a) <= 3: score += 7
    elif max(n_b, n_m, n_a) <= 4: score += 4
    else: score += 0
    # 4) Frequenza storica (max 12 pt) - somma freq normalizzata
    freq_score = sum(freq_map[x] for x in s) / (6 * max(freq_map.values()))
    score += freq_score * 12
    # 5) Ritardo medio composito (max 12 pt) - moderato, non estremo
    rit_avg = np.mean([rit_map[x] for x in s])
    # ottimo ~ ritardo moderato 15-40
    if 15 <= rit_avg <= 40: score += 12
    elif 10 <= rit_avg < 15 or 40 < rit_avg <= 50: score += 9
    elif rit_avg < 10 or rit_avg > 50: score += 5
    # 6) Diversita rispetto a estrazioni recenti (max 10 pt)
    overlap = len(set(s) & recent_nums_set)
    score += max(0, 10 - overlap * 2)
    # 7) Assenza pattern banali (max 12 pt)
    if not has_pattern_issues(s):
        score += 12
    # bonus se non ci sono consecutivi 3+
    consec = 1; max_consec = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]+1:
            consec += 1; max_consec = max(max_consec, consec)
        else:
            consec = 1
    if max_consec < 3: score += 4
    # 8) Stabilita temporale (max 10 pt)
    # presenza nelle ultime 36 estrazioni
    presenza_36 = sum(1 for x in s if freq_36_pct.get(x, 0) > 0)
    score += min(10, presenza_36 * 1.5)
    # 9) Valore decisionale (max 8 pt): no ripetizione completa con ultima estrazione
    overlap_last = len(set(s) & last_extraction_set)
    score += max(0, 8 - overlap_last * 4)
    # Normalizza max teorico ~ 102 -> riscalo su 100
    score_norm = min(100, round(score / 1.02, 1))
    return float(score_norm)

# Genera output finale
rows = []
for i, (strat, combo) in enumerate(combos):
    j, ss = pick_jolly_ss(combo, 9000+i)
    sc = score_combo(combo)
    rows.append({
        "strategia": strat,
        "combinazione": combo,
        "jolly": j,
        "superstar": ss,
        "score": sc,
        "n_even": sum(1 for x in combo if x % 2 == 0),
        "n_dec": len(set((x-1)//10 for x in combo)),
        "rit_avg": float(np.mean([rit_map[x] for x in combo])),
        "freq_avg": float(np.mean([freq_map[x] for x in combo])),
    })

out_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
out_df["rank"] = out_df.index + 1
out_df["combinazione_str"] = out_df["combinazione"].apply(lambda c: "-".join(f"{x:02d}" for x in sorted(c)))

# Export CSV
out_path = r"C:\Users\kyros\OneDrive\Desktop\superenalotto_dataset_package_v2\superenalotto_dataset_package_v2\combinazioni_ranking.csv"
out_df[["rank","strategia","combinazione_str","jolly","superstar","score","n_even","n_dec","rit_avg","freq_avg"]].to_csv(out_path, index=False)
print(f"\nRanking esportato in: {out_path}")
print("\n=== TOP 30 RANKING ===")
print(out_df[["rank","strategia","combinazione_str","jolly","superstar","score"]].to_string(index=False))

# Summary metrics globali da esportare per il report
summary = {
    "n_extractions": int(n_extractions),
    "date_min": date_min,
    "date_max": date_max,
    "n_even_pct": float(n_even/(n_even+n_odd)*100),
    "n_odd_pct": float(n_odd/(n_even+n_odd)*100),
    "decile_counts": dict(decile_counts),
    "fascia_counts": dict(fascia_counts),
    "chi2_stat": chi2_stat,
    "chi2_crit_05": chi2_critical_05,
    "chi2_crit_01": chi2_critical_01,
    "entropy_bits": entropy_bits,
    "entropy_max": entropy_max,
    "entropy_eff_pct": entropy_bits/entropy_max*100,
    "mc_p_value": p_value_empirical,
    "mc_mean": float(mc_chi2.mean()),
    "mc_std": float(mc_chi2.std()),
    "mc_p95": float(np.percentile(mc_chi2, 95)),
    "n_pair_expected": float(n_pair_expected),
    "n_triple_expected": float(n_triple_expected),
    "top10_freq": top10_freq.to_dict(orient="records"),
    "bottom10_freq": bottom10_freq.to_dict(orient="records"),
    "top_ritardatari": top_ritardatari.to_dict(orient="records"),
    "hot_numbers": hot_numbers.to_dict(orient="records"),
    "top_pairs": [(list(p), c) for p, c in top_pairs[:10]],
    "top_triples": [(list(t), c) for t, c in top_triples[:10]],
    "scaldati": scaldati,
    "raffreddati": raffreddati,
    "jolly_top": jolly_top,
    "ss_top": ss_top,
}
with open(r"C:\Users\kyros\OneDrive\Desktop\superenalotto_dataset_package_v2\superenalotto_dataset_package_v2\analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nSummary JSON salvato.")
