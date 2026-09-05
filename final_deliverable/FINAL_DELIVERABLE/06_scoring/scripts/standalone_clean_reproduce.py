from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT.parent


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    status = json.loads((PKG / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    mda_rows = rows(PKG / "MDA" / "mda_final_document_scores.csv")
    rating_rows = rows(PKG / "MDA" / "mda_final_ratings_long.csv")
    news_rows = rows(PKG / "NEWS" / "news_final_document_scores.csv")
    news_rating_rows = rows(PKG / "NEWS" / "news_final_ratings_long.csv")
    claim_links = rows(PKG / "CROSS_VALIDATION" / "claim_links.csv")
    if len(mda_rows) != int(status.get("mda_final_dataset_size", 0) or 0):
        raise SystemExit(f"mda row count mismatch: {len(mda_rows)}")
    if len(rating_rows) != len(mda_rows) * 5:
        raise SystemExit(f"ratings row count mismatch: {len(rating_rows)}")
    if len(news_rows) != int(status.get("news_final_dataset_size", 0) or 0):
        raise SystemExit(f"news row count mismatch: {len(news_rows)}")
    if len(news_rating_rows) != len(news_rows) * 5:
        raise SystemExit(f"news ratings row count mismatch: {len(news_rating_rows)}")
    if int(status.get("news_final_dataset_size", 0) or 0) == 0 and claim_links:
        raise SystemExit("claim_links must be empty when News is unavailable")
    bad_paths = []
    for path in PKG.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt", ".sh", ".py"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            blocked_user_path = "/Users/" + "zhaojiahao"
            if blocked_user_path in text:
                bad_paths.append(str(path.relative_to(PKG)))
    if bad_paths:
        raise SystemExit("absolute paths found: " + ", ".join(bad_paths[:5]))
    print(json.dumps({"standalone_reproduce": True, "mda_rows": len(mda_rows), "rating_rows": len(rating_rows), "news_rows": len(news_rows), "news_rating_rows": len(news_rating_rows), "claim_links": len(claim_links)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
