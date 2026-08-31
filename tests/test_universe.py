import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.universe import load_universe


class UniverseTests(unittest.TestCase):
    def test_merges_sp1500_without_overriding_curated_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            curated = root / "tickers.csv"
            sp1500 = root / "sp1500.csv"
            curated.write_text(
                "symbol,name,exchange\nAAA,Curated,NYSE\n",
                encoding="utf-8",
            )
            sp1500.write_text(
                "symbol,name,exchange,stage\n"
                "AAA,Index Name,US,1\n"
                "BBB,Added,US,1\n"
                "CCC,Later,US,2\n",
                encoding="utf-8",
            )
            with patch("src.universe.TICKERS_CSV", curated), patch(
                "src.universe.SP1500_CSV",
                sp1500,
            ):
                rows = load_universe()
        self.assertEqual(
            rows,
            [
                {"symbol": "AAA", "name": "Curated", "exchange": "NYSE"},
                {"symbol": "BBB", "name": "Added", "exchange": "US"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
