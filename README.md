# 📈 Stock Radar

Tägliches Research-Werkzeug: analysiert dein Aktienuniversum (Start: ~300, skalierbar auf 1500)
technisch und liefert jeden Tag die **Top 10 Daytrading-Chancen** und **Top 10 Langzeit-Einstiege** –
jeweils mit nachvollziehbarer Begründung und den passenden News-Headlines.

> ⚠️ **Keine Anlageberatung.** Alle Signale sind rein technisch berechnet. Entscheidungen triffst du selbst.

## Datenquelle & Kosten

- **Kurse:** Yahoo Finance über `yfinance` – kostenlos, ~15 Min verzögert.
- **News:** öffentliche Yahoo-Finance-RSS-Feeds pro Ticker (nur Headlines + Links).
- **Financial Times:** aus Lizenzgründen **nicht** automatisiert eingebunden. FT-Artikel liest du im
  eigenen Portal; öffentliche FT-Headlines lassen sich später als zusätzliche RSS-Quelle ergänzen.

## Ordnerstruktur

```
Stock-Radar/
├─ data/
│  ├─ tickers.csv          # DEIN Universum (symbol,name,exchange)
│  └─ output/latest.json   # Ergebnis der letzten Analyse
├─ src/
│  ├─ config.py            # Parameter (Indikator-Fenster, Top-N, Batchgröße)
│  ├─ universe.py          # Ticker laden
│  ├─ fetch.py             # Kurse holen (yfinance)
│  ├─ indicators.py        # RSI, MACD, ATR, Bollinger, SMA, Volumen …
│  ├─ score.py             # Scoring + deutsche Begründungen
│  ├─ news.py              # News-Headlines pro Ticker
│  └─ analyze.py           # Gesamt-Pipeline
├─ dashboard/app.py        # Streamlit-Dashboard
└─ .github/workflows/daily.yml  # automatische Läufe (mehrmals/Tag)
```

## Einmalige Einrichtung (Windows)

```powershell
cd "C:\Users\ththomas\OneDrive\Thomas Mercantile\Stock-Radar"
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Analyse laufen lassen

```powershell
.\.venv\Scripts\python.exe -m src.analyze
```

Erzeugt `data/output/latest.json` und gibt die Top-10-Listen in der Konsole aus.

## Dashboard öffnen

```powershell
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

Öffnet sich im Browser (http://localhost:8501).

## Eigene Ticker einpflegen

`data/tickers.csv` bearbeiten. Ein Yahoo-Symbol pro Zeile.
Suffixe: US = kein Suffix (`AAPL`), Xetra = `.DE` (`SAP.DE`), Euronext Amsterdam = `.AS`,
Paris = `.PA`, London = `.L`, SIX = `.SW`. Zeilen mit `#` werden ignoriert.

## Automatisierung (kostenlos)

Repo zu GitHub pushen → `.github/workflows/daily.yml` läuft an Handelstagen mehrmals,
aktualisiert `latest.json` und committet es. Optional das Dashboard gratis auf
Streamlit Community Cloud hosten (liest `latest.json` aus dem Repo).

## Parameter anpassen

Alles in `src/config.py`: `TOP_N`, Indikator-Fenster, `FETCH_CHUNK`.
Scoring-Logik und Begründungen in `src/score.py` – dort kannst du Gewichtungen ändern.
