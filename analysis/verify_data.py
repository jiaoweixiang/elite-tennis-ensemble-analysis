"""Check the deposited source workbooks and analysis tables against SHA-256."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "sha256.csv"


def files():
    yield DATA / "indicator_dictionary.csv"
    for directory in (DATA / "raw", DATA / "processed", DATA / "reference"):
        yield from sorted(p for p in directory.rglob("*") if p.is_file())


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Regenerate checksums after an audited data update")
    args = parser.parse_args()
    rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": hash_file(p), "bytes": p.stat().st_size}
            for p in files()]
    if args.write_manifest:
        with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "sha256", "bytes"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        with MANIFEST.open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        assert {r["path"]: (r["sha256"], int(r["bytes"])) for r in saved} == {
            r["path"]: (r["sha256"], r["bytes"]) for r in rows}

    train = pd.read_csv(DATA / "processed/train_grouped.csv")
    test = pd.read_csv(DATA / "processed/test_grouped.csv")
    dictionary = pd.read_csv(DATA / "indicator_dictionary.csv")
    assert len(dictionary) == 34
    assert dictionary.model_column.tolist() == list(train.columns[1:35])
    assert train.shape == (199, 43) and test.shape == (40, 43)
    assert train.match_id.nunique() == 59 and test.match_id.nunique() == 12
    assert (train["Set Result"] == 1).sum() == 163 and (test["Set Result"] == 1).sum() == 30
    assert 2022 not in set(train.year)
    assert set(test.year) == {2025}
    depth = ["Forecourt Short Balls", "Midcourt Balls", "Backcourt Deep Balls"]
    for table in (train, test):
        # Other source years retain ordinary two-decimal rounding (0.99/1.01).
        assert ((table[depth].sum(axis=1) - 1).abs() <= 0.011).all()
    assert ((train.loc[train.year == 2023, depth].sum(axis=1) - 1).abs() < 1e-9).all()
    source_2024 = next((DATA / "raw").glob("24年*.xlsx"))
    workbook = openpyxl.load_workbook(source_2024, read_only=True, data_only=True)
    assert workbook["Sheet1"]["Y26"].value == 155
    print(f"Verified {len(rows)} files, 239 sets and the 155 km/h source correction")


if __name__ == "__main__":
    main()
