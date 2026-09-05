# Methods Section

Status: Preliminary / not final.

## 1. Research design
This study constructs a reproducible text-scoring pipeline for Malaysian F&B firms. The scoring object is MD&A disclosure quality and news information quality, not a direct audit of firm truthfulness.

## 2. Sample construction
The sample is assembled from included MD&A registry records and real News registry records when available. Synthetic news placeholders are used only for pipeline testing and are excluded from final empirical analysis.

## 3. MD&A scoring object definition
The MD&A scoring object is the management discussion and analysis text. This is MD&A disclosure quality, not a full annual-report assessment.

## 4. News scoring object definition
News scoring evaluates news information quality and reliability at the document level.

## 5. Scoring dimensions
MD&A is scored on AR01-AR05 and News is scored on N01-N05. The dimension definitions and weights are fixed and are not changed by the paper output layer.

## 6. Weighting scheme
The main MD&A weights are AR01 0.15, AR02 0.25, AR03 0.30, AR04 0.20, AR05 0.10. The main News weights are N01 0.30, N02 0.25, N03 0.15, N04 0.20, N05 0.10.

## 7. Local LLM scoring setup
The scoring pipeline is designed for local Ollama with qwen3:8b using temperature=0, seed=42, num_ctx=4096, num_predict=192, think=false, stream=false, and JSON schema mode unless environment variables explicitly override these values and the runtime report records the actual values. If the local model is unavailable, the system writes blockers instead of fabricating scores.
qwen3:8b is a local 8B-scale model, not a human fact checker. News N01 factual accuracy evaluates internal consistency and verifiable cues inside the article text; it cannot prove that external facts are true.

## 8. Single-dimension scoring design
The stable design uses one LLM call per dimension and a minimal JSON object containing score, evidence, and reason. Python performs validation and aggregation.

## 9. AR02 numeric checker design
AR02 evaluates internal numeric consistency, calculation correctness, and logical traceability in the MD&A text. AR02 is not an external financial-statement audit and does not verify whether company financial statements are true.

## 10. Evidence locator and audit trail
Each dimension score must include an evidence locator such as MDA_Pxxx or NEWS_Pxxx. Documents with critical missing evidence cannot enter final outputs.

## 11. Quality control procedure
Outputs pass JSON parsing, schema validation, evidence checks, score range checks, review logging, and final freeze gates.

## 12. Statistical stability analysis
The package records weight sensitivity, repeatability, version stability, structure stability, and high-volatility cases when inputs exist.

## 13. Cross-validation design
Cross-validation is an auxiliary triangulation layer. It compares MD&A and News claims for support, contradiction, omission, overstatement, and repetition. Cross-validation does not alter MD&A or News scores.

## 14. Final freeze rule
The final freeze gate requires MD&A stability, real News scoring, valid schema status, low system error rate, and real News-backed claim links. When gates fail, the package is not paper-ready.
