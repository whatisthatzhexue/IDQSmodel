import csv
from pathlib import Path

from prepare_mda_inputs import build_document_id, paragraphize_mda_text, prepare_mda_inputs


def test_build_document_id_uses_required_mda_format():
    row = {"stock_code": "0012", "ticker": "3A", "report_year": "2024"}

    assert build_document_id(row) == "MDA_0012_3A_2024"


def test_paragraphize_mda_text_adds_numbered_mda_locators():
    text = "Management Discussion and Analysis\n\nRevenue increased by 20%.\nProfit improved."

    result = paragraphize_mda_text(text)

    assert "[MDA_P001]" in result
    assert "[MDA_P002]" in result
    assert "Revenue increased by 20%." in result


def test_prepare_mda_inputs_writes_registry_and_numbered_text(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    csv_path = source_dir / "final_ready_mda.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stock_code",
                "ticker",
                "company_name",
                "market",
                "sector",
                "report_year",
                "file_sha256",
                "detected_heading",
                "section_type_norm",
                "char_count",
                "codex_final_status",
                "mda_text",
                "final_pdf_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stock_code": "0012",
                "ticker": "3A",
                "company_name": "Three-A Resources Berhad",
                "market": "Main Market",
                "sector": "Food & Beverages",
                "report_year": "2024",
                "file_sha256": "abc123",
                "detected_heading": "MANAGEMENT DISCUSSION AND ANALYSIS",
                "section_type_norm": "STRICT_OR_STRONG_EQUIVALENT_MDA",
                "char_count": "120",
                "codex_final_status": "CODEX_FINAL_PASS",
                "mda_text": "MANAGEMENT DISCUSSION AND ANALYSIS\n\nRevenue increased by 20%.",
                "final_pdf_path": "/tmp/source.pdf",
            }
        )

    summary = prepare_mda_inputs(csv_path, tmp_path / "06_scoring")

    registry_path = tmp_path / "06_scoring" / "01_registry" / "mda_registry.csv"
    text_path = tmp_path / "06_scoring" / "02_extracted_text" / "mda" / "MDA_0012_3A_2024.txt"
    rows = list(csv.DictReader(registry_path.open(encoding="utf-8")))
    assert summary["included_rows"] == 1
    assert rows[0]["document_id"] == "MDA_0012_3A_2024"
    assert rows[0]["include_flag"] == "Yes"
    assert rows[0]["extraction_status"] == "success"
    assert rows[0]["text_path"] == "02_extracted_text/mda/MDA_0012_3A_2024.txt"
    assert text_path.read_text(encoding="utf-8").startswith("[MDA_P001]")
