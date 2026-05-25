# SuperEnalotto Dataset Package v2

Questa versione usa Lottologia come fonte tabellare stabile per estrazioni, Jolly e SuperStar.

## Installazione

```bash
pip install -r requirements.txt
```

## Esecuzione

```bash
python download_superenalotto_dataset.py --start-year 2025 --end-year 2026 --output data/superenalotto_2025_2026.csv
```

## Schema output

```csv
date,contest_number,n1,n2,n3,n4,n5,n6,jolly,superstar,winners_6,prize_6,source_url
```

## Nota

`winners_6` e `prize_6` restano vuoti perché Lottologia espone in questa pagina estrazioni + Jolly + SuperStar, non il dettaglio delle quote/vincitori del 6.
Per questi campi serve arricchimento dai dettagli concorso ufficiali Sisal/SuperEnalotto.
