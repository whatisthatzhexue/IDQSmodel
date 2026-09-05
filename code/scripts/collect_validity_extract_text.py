"""Extract full text from annual-report PDFs needed for the validity test (Size/Big4/Loss).

Usage (from ver2 dir):
  python code/scripts/collect_validity_extract_text.py --root .

Inputs:
  - ver2/08_reports/market_mapping.csv        (149 MDA company-year pairs)
  - verified_175_main_sample/*.pdf            (named {code}_{TICKER}_{year}_AnnualReport_partN_official.pdf)
Outputs:
  - ver2/12_validity/pdftotext/{basename}.txt (pdftotext -layout, UTF-8)
  - ver2/12_validity/extract_manifest.csv     (per-file status/pages/chars)
"""
import argparse
import csv
import glob
import os
import subprocess

PDFTOTEXT = r"C:\GitBash\mingw64\bin\pdftotext.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pdf-dir", default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    pdf_dir = os.path.abspath(args.pdf_dir) if args.pdf_dir else os.path.join(
        os.path.dirname(root), "verified_175_main_sample")
    out_dir = os.path.join(root, "12_validity", "pdftotext")
    os.makedirs(out_dir, exist_ok=True)

    mapping = os.path.join(root, "08_reports", "market_mapping.csv")
    rows = list(csv.reader(open(mapping, encoding="utf-8-sig")))[1:]
    needed = sorted({(r[1], r[3]) for r in rows})

    files = []
    for code, year in needed:
        hits = sorted(glob.glob(os.path.join(pdf_dir, f"{code}_*_{year}_AnnualReport_*_official.pdf")))
        files.extend(hits)
    print(f"PDFs to extract: {len(files)}", flush=True)

    manifest_path = os.path.join(root, "12_validity", "extract_manifest.csv")
    n_fail = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["pdf_file", "txt_file", "status", "pages", "chars"])
        for i, pdf in enumerate(files, 1):
            base = os.path.basename(pdf)[:-4]
            txt = os.path.join(out_dir, base + ".txt")
            if os.path.exists(txt) and os.path.getsize(txt) > 0:
                status, pages, chars = "cached", "", os.path.getsize(txt)
            else:
                r = subprocess.run(
                    [PDFTOTEXT, "-layout", "-enc", "UTF-8", pdf, txt],
                    capture_output=True, text=True)
                if r.returncode != 0 or not os.path.exists(txt) or os.path.getsize(txt) == 0:
                    status = f"fail: {(r.stderr or 'empty output')[:120]}"
                    n_fail += 1
                    pages, chars = "", 0
                else:
                    status, pages, chars = "ok", "", os.path.getsize(txt)
            if pages == "":
                try:
                    with open(txt, encoding="utf-8", errors="replace") as f:
                        pages = f.read().count("\f") + 1
                except Exception:
                    pages = ""
            w.writerow([os.path.basename(pdf), base + ".txt", status, pages, chars])
            if i % 20 == 0:
                print(f"[{i}/{len(files)}] extracted", flush=True)
    print(f"extraction done: {len(files) - n_fail} ok, {n_fail} failed -> {manifest_path}")


if __name__ == "__main__":
    main()
