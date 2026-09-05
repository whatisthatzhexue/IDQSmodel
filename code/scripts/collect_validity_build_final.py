"""Assemble the final validity-fundamentals CSV (with company names).

Usage (from ver2 dir):
  python code/scripts/collect_validity_build_final.py --root .

Inputs:
  - ver2/12_validity/validity_fundamentals_raw.csv
  - food&beverage company.xlsx (project root, 'main market' sheet)
Output:
  - ver2/12_validity/validity_fundamentals_final.csv
"""
import argparse
import csv
import os

import openpyxl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    names = {}
    xlsx = os.path.join(os.path.dirname(root), "food&beverage company.xlsx")
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    for row in wb["main market"].iter_rows(min_row=2, values_only=True):
        if row[0] and row[1]:
            names[str(row[1])] = str(row[0])

    raw = list(csv.reader(open(os.path.join(root, "12_validity", "validity_fundamentals_raw.csv"),
                              encoding="utf-8")))[1:]
    out = os.path.join(root, "12_validity", "validity_fundamentals_final.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["document_id", "stock_code", "ticker", "company_name", "year",
                    "auditor_firm", "big4", "auditor_method",
                    "total_assets_reported", "unit",
                    "profit_loss_reported", "loss_flag",
                    "ta_source_page", "pl_source_page", "notes"])
        for r in sorted(raw, key=lambda x: (x[1], int(x[3]))):
            doc_id, code, ticker, year = r[0], r[1], r[2], r[3]
            w.writerow([
                doc_id, code, ticker, names.get(code, ""), year,
                r[6], r[7], r[8],
                r[9], r[16],
                r[12], r[13],
                r[10], r[14], r[17],
            ])
    print(f"final CSV -> {out}")


if __name__ == "__main__":
    main()
