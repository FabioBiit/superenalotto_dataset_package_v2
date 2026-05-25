"""
Phase D - Dataset validator + enricher
Validates the extended 2009-2026 dataset and enriches it with derived metadata.

Output:
    data/superenalotto_full_history_validated.csv
    inference/phaseD_validation_report.json
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, time
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INF = ROOT / "inference"
INF.mkdir(exist_ok=True)

SRC = DATA / "superenalotto_2009-2026_extraction.csv"
OUT_CSV = DATA / "superenalotto_full_history_validated.csv"
OUT_JSON = INF / "phaseD_validation_report.json"

CP_FLUTTER = pd.Timestamp("2022-08-04")
CP_4WEEK = pd.Timestamp("2023-07-07")
CP_DM214 = pd.Timestamp("2026-01-27")

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
NUM_COLS = MAIN_COLS + ["jolly", "superstar"]


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df.columns = [c.strip().lower() for c in df.columns]
    if "super_star" in df.columns and "superstar" not in df.columns:
        df = df.rename(columns={"super_star": "superstar"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def schema_check(df: pd.DataFrame) -> dict:
    expected = ["date", "contest_number"] + NUM_COLS
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected + ["winners_6", "prize_6", "source_url"]]
    return {"missing": missing, "extra": extra, "n_rows": len(df), "n_cols": len(df.columns)}


def range_check(df: pd.DataFrame) -> dict:
    issues = {}
    for c in NUM_COLS:
        oor = df[(df[c] < 1) | (df[c] > 90)]
        if len(oor):
            issues[c] = {"n_out_of_range": int(len(oor)),
                         "sample_dates": oor["date"].dt.strftime("%Y-%m-%d").head(5).tolist()}
    return issues


def duplicate_within_draw(df: pd.DataFrame) -> dict:
    rows_with_dups = []
    jolly_clash = []
    for idx, row in df.iterrows():
        main = [row[c] for c in MAIN_COLS]
        if len(set(main)) != 6:
            rows_with_dups.append({"date": row["date"].strftime("%Y-%m-%d"),
                                   "contest": int(row["contest_number"]),
                                   "main": main})
        if row["jolly"] in main:
            jolly_clash.append({"date": row["date"].strftime("%Y-%m-%d"),
                                "contest": int(row["contest_number"]),
                                "jolly": int(row["jolly"]),
                                "main": main})
    return {"n_internal_duplicates_main": len(rows_with_dups),
            "sample_internal_duplicates": rows_with_dups[:5],
            "n_jolly_in_main": len(jolly_clash),
            "sample_jolly_clash": jolly_clash[:5]}


def date_checks(df: pd.DataFrame) -> dict:
    nan_dates = df["date"].isna().sum()
    dup_dates = df["date"].duplicated().sum()
    monotonic = df["date"].is_monotonic_increasing
    span_min = df["date"].min().strftime("%Y-%m-%d")
    span_max = df["date"].max().strftime("%Y-%m-%d")
    return {"n_nan_dates": int(nan_dates),
            "n_duplicate_dates": int(dup_dates),
            "monotonic_increasing": bool(monotonic),
            "span_min": span_min, "span_max": span_max}


def contest_checks(df: pd.DataFrame) -> dict:
    df["year"] = df["date"].dt.year
    yearly_resets = df.groupby("year")["contest_number"].agg(["min", "max", "count"]).to_dict("index")
    gaps_per_year = {}
    for y, g in df.groupby("year"):
        nums = sorted(g["contest_number"].unique())
        expected = list(range(min(nums), max(nums) + 1))
        missing = sorted(set(expected) - set(nums))
        if missing:
            gaps_per_year[int(y)] = missing[:10]
    return {"yearly_summary": {int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in yearly_resets.items()},
            "yearly_gaps": gaps_per_year}


def coverage_per_year(df: pd.DataFrame) -> dict:
    cov = df.groupby(df["date"].dt.year).size().to_dict()
    return {int(y): int(c) for y, c in cov.items()}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.day_name()
    df["hour"] = "20:00"
    df["operatore"] = np.where(df["date"] < CP_FLUTTER, "Sisal_Apax", "Sisal_Flutter")
    df["regime_4week"] = df["date"] >= CP_4WEEK
    df["regime_post_DM214"] = df["date"] >= CP_DM214
    return df


def main():
    print(f"[Phase D] Loading {SRC.name} ...")
    df = load()
    print(f"  rows={len(df)}, cols={list(df.columns)}")

    report = {
        "source_file": SRC.name,
        "n_rows": len(df),
        "schema": schema_check(df),
        "dates": date_checks(df),
        "range_violations": range_check(df),
        "draw_internal_consistency": duplicate_within_draw(df),
        "contest_numbers": contest_checks(df),
        "coverage_per_year": coverage_per_year(df),
    }

    # Severity
    severity = []
    if report["dates"]["n_nan_dates"] > 0:
        severity.append("CRITICAL: NaN dates present")
    if report["dates"]["n_duplicate_dates"] > 0:
        severity.append(f"WARN: {report['dates']['n_duplicate_dates']} duplicate dates")
    if report["range_violations"]:
        severity.append(f"CRITICAL: {sum(v['n_out_of_range'] for v in report['range_violations'].values())} out-of-range numbers")
    if report["draw_internal_consistency"]["n_internal_duplicates_main"] > 0:
        severity.append(f"CRITICAL: {report['draw_internal_consistency']['n_internal_duplicates_main']} draws with duplicate main numbers")
    if report["draw_internal_consistency"]["n_jolly_in_main"] > 0:
        severity.append(f"CRITICAL: {report['draw_internal_consistency']['n_jolly_in_main']} draws with jolly equal to a main number")
    if not severity:
        severity.append("OK: no critical issues detected (structural validation only)")
    report["severity"] = severity

    # Coverage summary
    cov = report["coverage_per_year"]
    print("\n[Coverage per year]")
    for y in sorted(cov.keys()):
        print(f"  {y}: {cov[y]} draws")

    print("\n[Severity]")
    for s in severity:
        print(f"  - {s}")

    # Enrichment
    print("\n[Enrichment] adding day_of_week, hour, operatore, regime_4week, regime_post_DM214 ...")
    df_enriched = enrich(df)

    # Save
    df_enriched.to_csv(OUT_CSV, index=False)
    print(f"\n[Output] enriched CSV -> {OUT_CSV.relative_to(ROOT)}")
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Output] validation report -> {OUT_JSON.relative_to(ROOT)}")

    # Print compact summary
    print(f"\n[Summary] {len(df_enriched)} draws, {df_enriched['date'].min().date()} -> {df_enriched['date'].max().date()}")
    op_counts = df_enriched["operatore"].value_counts().to_dict()
    print(f"  Operatore: {op_counts}")
    reg_4w = df_enriched["regime_4week"].sum()
    reg_dm = df_enriched["regime_post_DM214"].sum()
    print(f"  Regime 4-week: {reg_4w} draws | Post-DM214: {reg_dm} draws")


if __name__ == "__main__":
    main()
