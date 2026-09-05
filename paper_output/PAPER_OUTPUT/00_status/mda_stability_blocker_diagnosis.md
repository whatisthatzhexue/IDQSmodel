# MD&A Stability Blocker Diagnosis

## Success Rate Source
- numerator_success_count: 16
- denominator_selected_count: 27
- success_fraction: 16/27
- equals_16_over_27: true
- mda_stability_success_rate: 0.592593
- definition: numerator is selected MD&A stability-batch documents present in mda_v2_stability_document_scores; denominator is mda_v2_stability_batch_selection selected documents

## Failed Documents
- MDA_2828_CIHLDG_2024
- MDA_2836_CARLSBG_2022
- MDA_3026_DLADY_2021
- MDA_3026_DLADY_2022
- MDA_3026_DLADY_2023
- MDA_3689_FN_2020
- MDA_3689_FN_2021
- MDA_3689_FN_2022
- MDA_4065_PPB_2021
- MDA_4065_PPB_2022
- MDA_7103_SPRITZER_2023

## Failure Reason Distribution
- json_parse_failure: 11
- schema_validation_failure: 11
- evidence_locator_failure: 0
- missing_dimension: 55
- low_confidence: 5
- AR02_numeric_discrepancy: 11
- score_difference_too_large: 0
- repeatability_failure: 0
- version_comparison_failure: 0
- structure_comparison_failure: 1
- ollama_http_error: 0
- ollama_502_error: 0
- text_too_long: 0
- text_extraction_issue: 0
- missing_output: 0
- unknown: 0

## Top Failed Dimensions
- AR02: 27 cases
- ALL: 23 cases
- AR01: 11 cases
- AR03: 11 cases
- AR04: 11 cases
- AR05: 11 cases

## Top Failed Documents
- MDA_7103_SPRITZER_2023: 9 cases
- MDA_2828_CIHLDG_2024: 8 cases
- MDA_3026_DLADY_2022: 8 cases
- MDA_3689_FN_2022: 8 cases
- MDA_4065_PPB_2021: 8 cases
- MDA_3026_DLADY_2023: 7 cases
- MDA_3026_DLADY_2021: 7 cases
- MDA_2836_CARLSBG_2022: 7 cases
- MDA_3689_FN_2020: 7 cases
- MDA_3689_FN_2021: 7 cases

## Repairability
- parser_or_mapping_repairable_case_count: 33
- rescore_case_count: 55
- manual_review_case_count: 5

## Guardrails
- The 0.85 freeze threshold was not lowered.
- Failed samples were not deleted from the denominator.
- Cases CSV records actions without overwriting v1/v2/pilot/sample outputs.
