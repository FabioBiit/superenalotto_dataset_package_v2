#!/usr/bin/env python3
"""
wallenius_urn.py — Modello generativo dell'urna SuperEnalotto (Wallenius / Plackett-Luce)
=========================================================================================
"Ricreare la cesta": un'urna parametrica in cui ogni pallina i ha un peso omega_i
(la sua propensione effettiva di estrazione). Un'estrazione e' un campionamento
SENZA reimmissione proporzionale ai pesi -> processo di Wallenius (= Plackett-Luce),
lo stesso di cpp/src/models/wallenius.cpp, qui vettorizzato (Gumbel-top-k) e con il
layer di inferenza/potenza che il C++ non ha.

NON e' un predittore. Caratterizza la MACCHINA, non l'esito del singolo sorteggio.
Tre funzioni:
  1) SIMULARE storie sintetiche dell'urna                      -> WalleniusUrn.draw_history
  2) TESTARE H0: urna simmetrica (omega tutti uguali) con      -> test_symmetry
     null Monte Carlo corretto per il senza-reimmissione
  3) POWER ANALYSIS per simulazione: iniettare un bias noto e  -> power_analysis
     misurare quante volte la pipeline lo rileva (= floor di rilevabilita' EMPIRICO)

Perche' non predice: l'urna fisica e' caotica (Lyapunov > 0) e lo stato iniziale del
singolo sorteggio e' non osservabile. Una simulazione fedele riproduce la STATISTICA
(uniforme), non l'esito specifico. Vedi docs/official_mechanism.md.

Uso:
  python wallenius_urn.py                  # test simmetria + power analysis sul dataset validato
  python wallenius_urn.py --include-jolly  # usa anche la 7a pallina (stessa urna -> piu' dati)
  python wallenius_urn.py --no-power       # solo test di simmetria (veloce)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_BALLS = 90
K_MAIN = 6
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]


# --------------------------------------------------------------------------- #
#  Urna generativa (forward)
# --------------------------------------------------------------------------- #
class WalleniusUrn:
    """Urna di Wallenius: campionamento senza reimmissione proporzionale a omega.

    Implementato via Gumbel-top-k:
        key_i = log(omega_i) + Gumbel(0,1);  i k vincenti = i k key piu' alti.
    Equivale al campionamento sequenziale proporzionale ai pesi (Plackett-Luce),
    identico a Wallenius::sample_one del C++, ma vettorizzato su tutte le estrazioni.
    omega tutti uguali  ->  estrazione uniforme senza reimmissione (l'ipotesi nulla).
    """

    def __init__(self, omega: np.ndarray | None = None, n_balls: int = N_BALLS):
        if omega is None:
            omega = np.ones(n_balls)
        omega = np.asarray(omega, dtype=np.float64)
        if omega.shape != (n_balls,):
            raise ValueError(f"omega deve avere shape ({n_balls},)")
        if np.any(omega <= 0):
            raise ValueError("tutti i pesi omega devono essere > 0")
        self.n_balls = n_balls
        self.log_omega = np.log(omega)

    def draw_history(self, n_draws: int, k: int, rng: np.random.Generator) -> np.ndarray:
        """Matrice indicatrice (n_draws, n_balls) int8: 1 se la pallina e' estratta."""
        u = rng.random((n_draws, self.n_balls)) * (1.0 - 2e-12) + 1e-12
        gumbel = -np.log(-np.log(u))
        keys = self.log_omega[None, :] + gumbel
        topk = np.argpartition(-keys, kth=k - 1, axis=1)[:, :k]  # set, ordine irrilevante
        Y = np.zeros((n_draws, self.n_balls), dtype=np.int8)
        np.put_along_axis(Y, topk, 1, axis=1)
        return Y


# --------------------------------------------------------------------------- #
#  Statistiche
# --------------------------------------------------------------------------- #
def counts_of(Y: np.ndarray) -> np.ndarray:
    return Y.sum(axis=0).astype(np.float64)


def g_statistic(counts: np.ndarray) -> float:
    """G-test (likelihood ratio) della uniformita' dei conteggi per pallina."""
    total = counts.sum()
    expected = total / len(counts)
    nz = counts > 0
    return float(2.0 * np.sum(counts[nz] * np.log(counts[nz] / expected)))


def fit_weights(Y: np.ndarray) -> np.ndarray:
    """Stima al prim'ordine dei pesi: omega_i ~ conteggio_i (normalizzati a media 1).

    Per pesi vicini all'uniforme E[conteggio_i] ~ omega_i a meno di O((omega-1)^2);
    la MLE esatta Plackett-Luce raffinerebbe questo, ma nel regime quasi-uniforme la
    differenza e' trascurabile. Serve a *riportare* il bias inferito, non al test.
    """
    cnt = counts_of(Y)
    return np.clip(cnt / cnt.mean(), 1e-6, None)


def bh_reject(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: ritorna maschera bool dei rifiuti (controllo FDR)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    rej = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = int(np.max(np.where(passed)[0]))
        rej[order[: kmax + 1]] = True
    return rej


def mc_null_g(n_draws: int, k: int, n_sim: int, rng: np.random.Generator) -> np.ndarray:
    """Distribuzione nulla di G sotto urna simmetrica (omega uguali), stessi N e k.
    Cattura la struttura SENZA reimmissione, che il chi^2 analitico ignora."""
    urn = WalleniusUrn(np.ones(N_BALLS))
    out = np.empty(n_sim)
    for s in range(n_sim):
        out[s] = g_statistic(counts_of(urn.draw_history(n_draws, k, rng)))
    return out


# --------------------------------------------------------------------------- #
#  Test di simmetria (inferenza inversa)
# --------------------------------------------------------------------------- #
def test_symmetry(Y, k=K_MAIN, n_sim=2000, rng=None, null_g=None):
    if rng is None:
        rng = np.random.default_rng(0)
    N = Y.shape[0]
    cnt = counts_of(Y)
    g_obs = g_statistic(cnt)
    df = N_BALLS - 1
    chi2_p = float(stats.chi2.sf(g_obs, df))                  # null analitico (CON reimmissione)
    if null_g is None:
        null_g = mc_null_g(N, k, n_sim, rng)
    mc_p = float((1 + np.sum(null_g >= g_obs)) / (1 + len(null_g)))   # null SENZA reimmissione

    # deviazioni per-pallina (descrittive)
    p0 = k / N_BALLS
    exp = N * p0
    sd = np.sqrt(N * p0 * (1.0 - p0))
    z = (cnt - exp) / sd
    per_p = 2.0 * stats.norm.sf(np.abs(z))
    rej = bh_reject(per_p, alpha=0.05)
    order = np.argsort(-np.abs(z))
    extremes = [
        {"numero": int(i + 1), "conteggio": int(cnt[i]), "atteso": round(float(exp), 1),
         "z": round(float(z[i]), 2), "signif_BH": bool(rej[i])}
        for i in order[:6]
    ]
    return {
        "n_draws": int(N), "k": int(k),
        "G_obs": round(g_obs, 2), "df": df,
        "chi2_p_analitico": round(chi2_p, 4),
        "mc_p_value": round(mc_p, 4),
        "null_G_mean": round(float(null_g.mean()), 2),
        "null_G_q95": round(float(np.quantile(null_g, 0.95)), 2),
        "n_balls_signif_BH": int(rej.sum()),
        "estremi": extremes,
        "verdetto": ("Compatibile con urna simmetrica: nessuna deviazione robusta."
                     if mc_p > 0.05 else
                     "Deviazione dalla simmetria rilevata: indagare/raffinare."),
    }


# --------------------------------------------------------------------------- #
#  Power analysis per simulazione
# --------------------------------------------------------------------------- #
def power_analysis(n_draws, bias_factors, k=K_MAIN, n_rep=400, n_null=2000,
                   alpha=0.05, rng=None, target=0):
    """Inietta omega[target]=bias_factor, misura quanto spesso la pipeline lo rileva.

    Due nozioni di potenza:
      - test globale G  : rileva 'l'urna NON e' simmetrica' (qualsiasi deviazione)
      - rileva pallina  : flagga proprio la pallina sbilanciata (BH su 90, alpha)
    """
    if rng is None:
        rng = np.random.default_rng(123)
    null_g = mc_null_g(n_draws, k, n_null, rng)
    thr = float(np.quantile(null_g, 1.0 - alpha))            # soglia test globale G

    p0 = k / N_BALLS
    exp = n_draws * p0
    sd = np.sqrt(n_draws * p0 * (1.0 - p0))

    rows = []
    for bf in bias_factors:
        omega = np.ones(N_BALLS)
        omega[target] = bf
        urn = WalleniusUrn(omega)
        hit_global = 0
        hit_ball = 0
        for _ in range(n_rep):
            cnt = counts_of(urn.draw_history(n_draws, k, rng))
            if g_statistic(cnt) >= thr:
                hit_global += 1
            z = (cnt - exp) / sd
            per_p = 2.0 * stats.norm.sf(np.abs(z))
            if bh_reject(per_p, alpha=alpha)[target]:
                hit_ball += 1
        rows.append({
            "bias_factor": round(float(bf), 3),
            "bias_pct": round((bf - 1.0) * 100.0, 1),
            "power_test_globale_G": round(hit_global / n_rep, 3),
            "power_rileva_pallina_BH": round(hit_ball / n_rep, 3),
        })
    return {"n_draws": int(n_draws), "alpha": alpha, "soglia_G_globale": round(thr, 2),
            "n_rep": n_rep, "n_null": n_null, "tabella": rows}


# --------------------------------------------------------------------------- #
#  Caricamento dati
# --------------------------------------------------------------------------- #
def load_draws(csv_path: str, include_jolly: bool = False):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["date", "contest_number"]).reset_index(drop=True)
    cols = MAIN_COLS + (["jolly"] if include_jolly else [])
    k = len(cols)
    N = len(df)
    vals = df[cols].to_numpy(dtype=int)
    Y = np.zeros((N, N_BALLS), dtype=np.int8)
    for c in range(k):
        Y[np.arange(N), vals[:, c] - 1] = 1
    if not (Y.sum(axis=1) == k).all():
        bad = int(np.sum(Y.sum(axis=1) != k))
        raise ValueError(f"{bad} estrazioni con numeri duplicati o fuori range 1-90")
    return Y, k, df


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Urna Wallenius: simula, testa simmetria, power analysis.")
    ap.add_argument("--csv", default=str(repo / "data" / "superenalotto_full_history_validated.csv"))
    ap.add_argument("--include-jolly", action="store_true", help="usa la 7a pallina (stessa urna)")
    ap.add_argument("--n-sim", type=int, default=2000, help="iterazioni null Monte Carlo")
    ap.add_argument("--n-rep", type=int, default=400, help="ripetizioni per la power analysis")
    ap.add_argument("--no-power", action="store_true", help="salta la power analysis")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(repo / "inference"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    Y, k, df = load_draws(args.csv, args.include_jolly)
    label = "6 principali + jolly" if args.include_jolly else "6 principali"
    print(f"[LOAD] {Y.shape[0]} estrazioni | pick k={k} ({label})")
    print(f"       {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")

    print("\n[TEST] Simmetria dell'urna  (H0: tutti i pesi omega uguali)")
    res = test_symmetry(Y, k=k, n_sim=args.n_sim, rng=rng)
    print(f"  G_obs = {res['G_obs']}   df = {res['df']}")
    print(f"  p chi^2 analitico (assume reimmissione)      = {res['chi2_p_analitico']}")
    print(f"  p Monte Carlo (senza reimmissione, CORRETTO) = {res['mc_p_value']}   <==")
    print(f"  null G: media={res['null_G_mean']}  q95={res['null_G_q95']}")
    print(f"  palline significative dopo BH(0.05): {res['n_balls_signif_BH']}/90")
    print("  estremi per |z|:")
    for e in res["estremi"]:
        flag = "  <-- SIGNIF" if e["signif_BH"] else ""
        print(f"    n.{e['numero']:>2}  cnt={e['conteggio']:>4}  atteso={e['atteso']:>6}  z={e['z']:+.2f}{flag}")
    print(f"  => {res['verdetto']}")

    out = {"dataset": Path(args.csv).name, "include_jolly": args.include_jolly,
           "symmetry_test": res}

    if not args.no_power:
        print("\n[POWER] Quanto bias su UNA pallina servirebbe per rilevarlo? (simulazione)")
        bias = [1.05, 1.10, 1.15, 1.20, 1.30, 1.50]
        for N in sorted({Y.shape[0], 1133}):
            pa = power_analysis(N, bias, k=k, n_rep=args.n_rep, n_null=args.n_sim, rng=rng)
            print(f"\n  N={N}  (soglia G a 0.05 = {pa['soglia_G_globale']}, n_rep={pa['n_rep']})")
            print(f"    {'bias%':>6} | {'potenza test globale':>20} | {'potenza rileva pallina(BH)':>26}")
            print(f"    {'-'*6} | {'-'*20} | {'-'*26}")
            for r in pa["tabella"]:
                print(f"    {r['bias_pct']:>5}% | {r['power_test_globale_G']:>20} | {r['power_rileva_pallina_BH']:>26}")
            mde = next((r["bias_pct"] for r in pa["tabella"] if r["power_rileva_pallina_BH"] >= 0.8), None)
            print(f"    => MDE(80%, rileva la pallina, BH) {'~ ' + str(mde) + '%' if mde else '> 50% (oltre il range testato)'}")
            out[f"power_N{N}"] = pa

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    fp = outdir / ("wallenius_urn_report" + ("_jolly" if args.include_jolly else "") + ".json")
    fp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] report -> {fp}")
    print("\nNota: questo modulo CARATTERIZZA l'urna (quanto e' simmetrica); non predice le estrazioni.")


if __name__ == "__main__":
    main()
