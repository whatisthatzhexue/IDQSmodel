from validate_model_scores import normalize_scoring_payload, validate_scoring_payload


def _valid_payload():
    return {
        "document_id": "MDA_1234_TEST_2024",
        "doc_type": "mda",
        "scoring_basis": "mda_disclosure_quality",
        "scorebook_version": "2026-06-15",
        "evidence_limitation": "Based only on provided MD&A text.",
        "dimension_scores": [
            {
                "dimension_code": code,
                "dimension_name": name,
                "raw_score": 3,
                "reason": "Basic coverage with traceable evidence.",
                "evidence_locator": "[MDA_P001]",
                "evidence_text_short": "Revenue increased by 20%.",
                "confidence_level": "medium",
                "missing_type": "none",
                "comment_short": "Acceptable disclosure.",
            }
            for code, name in [
                ("AR01", "MD&A content completeness"),
                ("AR02", "MD&A internal numeric consistency"),
                ("AR03", "Operating analysis specificity"),
                ("AR04", "Risk and forward-looking disclosure"),
                ("AR05", "Readability and structure"),
            ]
        ],
        "flags": {
            "flag_unverified_key_number": 0,
            "flag_excessive_promotional_tone": 0,
            "flag_nonstandard_audit": 0,
            "flag_disclosure_replacement": 0,
            "flag_retraction_or_correction_needed": 0,
        },
        "overall_comment": "The MD&A is adequate.",
        "needs_review": False,
        "review_reasons": [],
    }


def test_valid_mda_payload_passes_schema_and_paragraph_locator_check():
    result = validate_scoring_payload(_valid_payload(), "[MDA_P001]\nRevenue increased by 20%.")

    assert result.is_valid is True
    assert result.issues == []


def test_schema_rejects_wrong_dimension_count_and_missing_evidence():
    payload = _valid_payload()
    payload["dimension_scores"] = payload["dimension_scores"][:4]

    result = validate_scoring_payload(payload, "[MDA_P001]\nRevenue increased by 20%.")

    assert result.is_valid is False
    assert "dimension_count_not_5" in result.issues


def test_mda_scope_violation_is_detected():
    payload = _valid_payload()
    payload["dimension_scores"][1]["reason"] = "The numbers match the audited financial statements."

    result = validate_scoring_payload(payload, "[MDA_P001]\nRevenue increased by 20%.")

    assert result.is_valid is False
    assert "mda_scope_violation_audited_financial_statement_match" in result.issues


def test_raw_score_one_with_positive_reason_is_flagged_as_contradiction():
    payload = _valid_payload()
    payload["dimension_scores"][0]["raw_score"] = 1
    payload["dimension_scores"][0]["reason"] = "The text covers business overview, financial review, risks, and outlook comprehensively."

    result = validate_scoring_payload(payload, "[MDA_P001]\nRevenue increased by 20%.")

    assert result.is_valid is False
    assert "raw_score_reason_contradiction" in result.issues


def test_raw_score_one_with_negative_specificity_reason_is_not_flagged():
    payload = _valid_payload()
    payload["dimension_scores"][2]["raw_score"] = 1
    payload["dimension_scores"][2]["reason"] = "No operating analysis specificity, focuses on product launches and certifications."
    payload["dimension_scores"][2]["comment_short"] = "No operating analysis specificity."

    result = validate_scoring_payload(payload, "[MDA_P001]\nRevenue increased by 20%.")

    assert result.is_valid is True
    assert "raw_score_reason_contradiction" not in result.issues


def test_raw_score_one_with_negative_reason_and_positive_quote_is_not_flagged():
    payload = _valid_payload()
    payload["dimension_scores"][4]["raw_score"] = 1
    payload["dimension_scores"][4]["reason"] = "Lacks financial analysis, focuses on branding and marketing."
    payload["dimension_scores"][4]["comment_short"] = "Lacks financial analysis, focuses on branding and marketing."
    payload["dimension_scores"][4]["evidence_text_short"] = "Creating Good Times, celebrating a brand anniversary."

    result = validate_scoring_payload(payload, "[MDA_P001]\nCreating Good Times, celebrating a brand anniversary.")

    assert result.is_valid is True
    assert "raw_score_reason_contradiction" not in result.issues


def test_normalize_scoring_payload_wraps_bare_evidence_locator_tokens():
    payload = _valid_payload()
    payload["dimension_scores"][0]["evidence_locator"] = "MDA_P001, MDA_P002"

    normalized = normalize_scoring_payload(payload, "mda", "MDA_1234_TEST_2024")

    assert normalized["scoring_basis"] == "mda_disclosure_quality"
    assert normalized["scorebook_version"] == "2026-06-15"
    assert normalized["dimension_scores"][0]["evidence_locator"] == "[MDA_P001], [MDA_P002]"
