#!/usr/bin/env python3
"""
Scarica e normalizza l'archivio SuperEnalotto da Lottologia.
Output schema:
date,contest_number,n1,n2,n3,n4,n5,n6,jolly,superstar,winners_6,prize_6,source_url

Nota: Lottologia espone estrazioni + Jolly + SuperStar, ma non quote/vincitori del 6.
Per questo winners_6 e prize_6 restano vuoti.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://lottologia.com/superenalotto/estrazioni"

MONTHS = {
    "gen": 1, "gennaio": 1,
    "feb": 2, "febbraio": 2,
    "mar": 3, "marzo": 3,
    "apr": 4, "aprile": 4,
    "mag": 5, "maggio": 5,
    "giu": 6, "giugno": 6,
    "lug": 7, "luglio": 7,
    "ago": 8, "agosto": 8,
    "set": 9, "settembre": 9,
    "ott": 10, "ottobre": 10,
    "nov": 11, "novembre": 11,
    "dic": 12, "dicembre": 12,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}

@dataclass
class Draw:
    date: str
    contest_number: str
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    n6: int
    jolly: int
    superstar: int
    winners_6: str
    prize_6: str
    source_url: str


def build_url(year: int) -> str:
    # Pagina corrente: /estrazioni per 2026; pagine storiche: /estrazioni/{year}
    current_year = datetime.now().year
    return BASE_URL if year == current_year else f"{BASE_URL}/{year}"


def parse_italian_date(value: str, fallback_year: int) -> str:
    parts = value.strip().lower().split()
    if len(parts) != 3:
        raise ValueError(f"Data non riconosciuta: {value!r}")

    day = int(parts[0])
    month_token = parts[1].strip(".")
    year = int(parts[2]) if parts[2].isdigit() else fallback_year

    month = MONTHS.get(month_token)
    if not month:
        raise ValueError(f"Mese non riconosciuto: {month_token!r}")

    return datetime(year, month, day).date().isoformat()


def fetch_html(url: str, timeout: int = 30, retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            wait = 2 * attempt
            print(f"WARN tentativo {attempt}/{retries} fallito per {url}: {exc}", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"Impossibile scaricare {url}: {last_error}")


def parse_draws_from_lottologia(html: str, year: int, source_url: str) -> list[Draw]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

    # Normalizza spazi e righe vuote, preservando la separazione per pattern multilinea.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    date_pattern = r"(\d{1,2}\s+(?:Gen|Feb|Mar|Apr|Mag|Giu|Lug|Ago|Set|Ott|Nov|Dic)\s+\d{4})"

    pattern = re.compile(
        date_pattern
        + r"\s*\n\s*Numeri\s*\n\s*"
        + r"((?:\d{1,2}\s+){5}\d{1,2})"
        + r"\s*\n\s*jolly\s*\n\s*(\d{1,2})"
        + r"\s*\n\s*superstar\s*\n\s*(\d{1,2})",
        re.IGNORECASE,
    )

    draws: list[Draw] = []
    for idx, match in enumerate(pattern.finditer(clean_text), start=1):
        raw_date, raw_numbers, raw_jolly, raw_superstar = match.groups()
        numbers = [int(n) for n in re.findall(r"\d{1,2}", raw_numbers)]

        if len(numbers) != 6:
            continue

        # Lottologia non espone il numero concorso in questo archivio.
        # Usiamo progressivo provvisorio entro anno in ordine cronologico dopo il sort finale.
        draws.append(
            Draw(
                date=parse_italian_date(raw_date, year),
                contest_number="",
                n1=numbers[0],
                n2=numbers[1],
                n3=numbers[2],
                n4=numbers[3],
                n5=numbers[4],
                n6=numbers[5],
                jolly=int(raw_jolly),
                superstar=int(raw_superstar),
                winners_6="",
                prize_6="",
                source_url=source_url,
            )
        )

    # Imposta contest_number progressivo per anno, dal più vecchio al più recente.
    draws = sorted(draws, key=lambda d: d.date)
    for i, draw in enumerate(draws, start=1):
        draw.contest_number = str(i)

    return draws


def deduplicate(draws: Iterable[Draw]) -> list[Draw]:
    seen: set[tuple[str, int, int, int, int, int, int]] = set()
    result: list[Draw] = []

    for draw in draws:
        key = (draw.date, draw.n1, draw.n2, draw.n3, draw.n4, draw.n5, draw.n6)
        if key not in seen:
            seen.add(key)
            result.append(draw)

    return sorted(result, key=lambda d: d.date)


def write_csv(draws: list[Draw], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "date", "contest_number", "n1", "n2", "n3", "n4", "n5", "n6",
        "jolly", "superstar", "winners_6", "prize_6", "source_url"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(draw) for draw in draws)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/superenalotto_2025_2026.csv"))
    parser.add_argument("--sleep", type=float, default=1.0, help="Pausa tra richieste HTTP")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("Errore: --start-year non può essere maggiore di --end-year")

    all_draws: list[Draw] = []

    for year in range(args.start_year, args.end_year + 1):
        url = build_url(year)
        html = fetch_html(url)
        draws = parse_draws_from_lottologia(html, year, url)
        print(f"OK {url} -> {len(draws)} estrazioni")
        all_draws.extend(draws)
        time.sleep(args.sleep)

    all_draws = deduplicate(all_draws)
    write_csv(all_draws, args.output)

    print(f"\nCSV creato: {args.output}")
    print(f"Totale estrazioni: {len(all_draws)}")
    print("Nota: winners_6 e prize_6 sono lasciati vuoti perché non presenti nell'archivio Lottologia.")


if __name__ == "__main__":
    main()
