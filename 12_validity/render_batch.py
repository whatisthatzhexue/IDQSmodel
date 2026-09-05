"""Render candidate statement pages of given PDFs to PNG for Windows OCR.
Usage: python render_batch.py  (edit JOBS below)"""
import pymupdf

BASE = "E:/MyData/大工/2026/CIS/Research_v2/verified_175_main_sample"
JOBS = [
    # (tag, pdf_name, heading_text, extra_pages_after_heading, pages_before)
    ("CARLSBG2021_SoPL", "2836_CARLSBG_2021_AnnualReport_part1_official.pdf",
     "PROFIT OR LOSS", 3, 1),
    ("CARLSBG2021_SoFP", "2836_CARLSBG_2021_AnnualReport_part1_official.pdf",
     "STATEMENTS OF FINANCIAL POSITION", 1, 0),
    ("FN2021_SoPL", "3689_FN_2021_AnnualReport_part2_official.pdf",
     "PROFIT OR LOSS", 2, 0),
    ("HBGLOB2020_SoFP", "5187_HBGLOB_2020_AnnualReport_part1_official.pdf",
     "STATEMENTS OF FINANCIAL POSITION", 2, 0),
]


def main():
    for tag, pdf, hdr, ahead, before in JOBS:
        d = pymupdf.open(f"{BASE}/{pdf}")
        found = [i for i in range(len(d)) if hdr in d[i].get_text().upper()]
        print(f"{tag}: {len(d)} pages, heading pages: {found[:6]}")
        if not found:
            continue
        start = found[-1] - before  # last occurrence = real statement
        for k in range(-before, ahead + 1):
            i = start + k
            if 0 <= i < len(d):
                pix = d[i].get_pixmap(dpi=300)
                pix.save(f"E:/MyData/大工/2026/CIS/Research_v2/ver2/12_validity/render/{tag}_p{i+1}.png")


if __name__ == "__main__":
    main()
