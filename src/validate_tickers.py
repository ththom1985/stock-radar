"""One-off helper: check which symbols in tickers.csv actually return data.
Uses the same batch fetch path as the real pipeline.
Run:  python -m src.validate_tickers
"""
from .universe import load_universe
from .fetch import fetch_prices


def main():
    uni = load_universe()
    symbols = [u["symbol"] for u in uni]
    names = {u["symbol"]: u["name"] for u in uni}
    prices = fetch_prices(symbols)

    ok, fail = [], []
    for s in symbols:
        df = prices.get(s)
        if df is not None and "Close" in df.columns and df["Close"].dropna().shape[0] > 0:
            ok.append((s, names[s], float(df["Close"].dropna().iloc[-1])))
        else:
            fail.append((s, names[s]))

    print(f"\n=== OK ({len(ok)}/{len(symbols)}) ===")
    for s, name, last in ok:
        print(f"  {s:<10} {last:>10.2f}  {name}")
    print(f"\n=== KEINE DATEN ({len(fail)}) ===")
    for s, name in fail:
        print(f"  {s:<10} {name}")


if __name__ == "__main__":
    main()
