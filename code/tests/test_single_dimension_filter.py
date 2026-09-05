from __future__ import annotations

from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import REGISTRY_HEADERS, write_csv_rows


def test_run_single_dimension_scoring_filters_dimension_codes(tmp_path):
    root = tmp_path / "06_scoring"
    text_path = root / "02_extracted_text" / "mda" / "MDA_A.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("[MDA_P001]\nRevenue increased by 10% due to demand and cost control.\n", encoding="utf-8")
    row = {field: "" for field in REGISTRY_HEADERS["mda"]}
    row.update(
        {
            "document_id": "MDA_A",
            "doc_type": "mda",
            "include_flag": "Yes",
            "ticker": "AAA",
            "company_name": "Alpha",
            "report_year": "2024",
            "text_path": "02_extracted_text/mda/MDA_A.txt",
        }
    )
    write_csv_rows(root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], [row])

    summary = run_single_dimension_scoring(
        "mda",
        root=root,
        output_name="dimension_filter",
        ratings_prefix="dimension_filter",
        mock=True,
        dimension_codes=["AR02"],
    )
    parsed_files = sorted((root / "06_ratings" / "dimension_filter" / "per_dimension").glob("*_parsed.json"))

    assert summary["dimension_success_count"] == 1
    assert [path.name for path in parsed_files] == ["MDA_A_AR02_parsed.json"]
