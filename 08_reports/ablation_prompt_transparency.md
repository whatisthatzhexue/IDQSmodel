# 消融实验提示词透明性说明

> 更新：2026-08-20
> 目的：让消融实验（Prompt 扰动 / 鲁棒性检验）的提示词变化完全可追溯、可复现。

## 一、提示词结构（三层）

| 层 | 位置 | 内容 |
|----|------|------|
| **基座提示词**（生产评分同款） | `code/scripts/run_single_dimension_scoring.py` 的 `_single_dimension_prompt` / `_single_dimension_prompt_parts`（约 380-436 行） | 评分任务说明、文档元数据、维度定义、数值校验摘要 |
| **系统提示词**（LLM 客户端层） | `code/scripts/llm_clients.py` `OllamaClient.score_document` → system message | "You are a strict JSON-only scoring assistant. Follow the research scope exactly." |
| **扰动注入点**（消融唯一变动） | `code/scripts/run_ablation_variant_scoring.py` 的 `TONE_INSTRUCTIONS`（约 50-59 行）+ `build_variant_prompt`（约 62-100 行） | 在 "Return exactly one JSON object…" 指令之后、示例行之前插入一段语气指令 |

## 二、扰动内容（唯一被改变的部分）

**极端指令扰动**（显式要求改分方向）：
```
strict:  Scoring tone instruction: Be STRICT. Only award high scores when the text provides
         explicit, concrete, verifiable evidence for the dimension. When in doubt, score LOWER.
lenient: Scoring tone instruction: Be LENIENT. Focus on overall information quality rather than
         specific missing details. When in doubt, score HIGHER.
```

**温和措辞扰动**（无方向命令，仅换描述性标准）：
```
mild_strict:  Scoring tone instruction: Apply a rigorous, conservative evaluation standard. High
              scores require clear and explicit evidence in the text; indirect or ambiguous support
              should not be over-credited.
mild_lenient: Scoring tone instruction: Apply a constructive, holistic evaluation standard. Credit
              substantive discussion and overall disclosure quality even when some specific details
              are missing.
```

除上述语气指令外，**其余提示词与生产评分逐字相同**（`temperature=0, seed=42, num_ctx=4096, model=qwen3:8b` 均不变）。温和变体用于回答"现实性措辞变化是否影响评分"，与极端变体（"指令遵循度"）区分开。

## 三、执行与鲁棒性保障

| 机制 | 说明 |
|------|------|
| 重试 | 每条评分最多 2 次尝试；JSON 解析失败自动重试 |
| 兜底 | 两次均失败时记录 `fallback`（score=3, confidence_level=low），**不中断整批任务** |
| 断点续跑 | 每维度结果缓存于 `06_ratings/mda_ablation_{tone}/per_dimension/{doc_id}_{dim}_parsed.json`；重跑脚本自动跳过已成功条目 |
| 原始证据 | 每次尝试的原始模型输出存 `05_raw_model_outputs/mda_ablation_{tone}/{doc_id}_{dim}_attempt_{N}.json`（含 content 与解析结果） |
| 溯源标记 | 评分长表 `review_round="ablation"`，`reviewer_id="local_qwen3_8b_{tone}"`，可与生产评分行（review_round="initial"）区分 |

## 四、产物路径

- 评分：`ver2/06_ratings/mda_ablation_strict/`、`ver2/06_ratings/mda_ablation_lenient/`
  - `{tone}_ratings_long.csv`（维度级）、`{tone}_document_scores.csv`（文档级）、`{tone}_summary.json`
- 显著性检验与报告：`code/scripts/run_ablation_experiment.py` → `ver2/08_reports/ablation_report.md`
- 样本：`ver2/08_reports/ablation_sample.csv`（25 个文档）

## 五、修复记录（2026-08-20）

1. **API 适配**：`run_ablation_variant_scoring.py` 原引用的旧版 API 已被队友重构（`_dimension_schema` → `validate_json_outputs.simple_dimension_schema`、`_build_rating_rows`/`_build_document_score_row` 迁至 `run_scoring.py` 且签名变化）。改用新 API 重建评分行，保留消融溯源标记（review_round/reviewer_id 覆写）。
2. **提示词对齐生产管线**：消融提示词现由 `_single_dimension_prompt_parts`（生产同款）生成，仅注入语气行（注入点固定：`Valid response example:` 之前）。评分上下文使用生产管线预建的维度级上下文文件 `02_extracted_text/mda_dimension_context/{doc_id}_{dim}.txt`（解决长 MDA 超 4096 上下文截断导致的胡答）。
3. **重试对齐**：第 2 次尝试追加生产同款纠正语 `Retry: return exactly one JSON object with only score, evidence, reason.`（qwen3:8b 偶发返回 `reasoning` 字段导致校验失败）。
4. **基线可比性**：基线（repeatability run_a）原始输出含 `context_path/prompt_budget/fallback_used` 字段，确认与当前生产管线同代——基线 vs 变体的差异仅来自语气行。
5. **p 值口径修正（2026-08-24）**：`paired_t_test` 原用正态近似计算双尾 p（n=25 时系统性偏小）；改用 t 分布（df=n−1=24）。新 p 值 strict 0.0052 / lenient <0.0001 / mild_strict 0.361 / mild_lenient 0.048，显著性判定不变。

## 六、实验结果摘要（详见 ablation_report.md）

| 变体 | 类型 | 总分均差 | 配对 T | p | 等级稳定率 | 维度一致率 |
|------|------|---------|--------|-----|-----------|-----------|
| strict | 极端指令 | -7.95 | -3.072 | 0.0052** | 56.0% | 46.4% |
| lenient | 极端指令 | +19.20 | 5.936 | <0.0001** | 24.0% | 40.0% |
| mild_strict | 温和措辞 | -2.65 | -0.932 | 0.361 | 48.0% | 47.2% |
| mild_lenient | 温和措辞 | +5.35 | 2.085 | 0.048* | 48.0% | 43.2% |

- 极端指令方向符合预期（strict 降低、lenient 提高），且差异显著——记录的是"模型遵循指令"的敏感性。
- 温和措辞：strict 方向**不显著**（p=0.361）；lenient 方向弱显著（p=0.048，接近 0.05 边界，幅度 +5.35 远小于极端 +19.20）。
- 等级稳定率在**所有**变体（含不显著变体）均低于 90%：25 个样本基线总分集中在等级边界（均值 56.8，D/C 界 55），小幅均值偏移也会翻档——90% 指标对阈值放置更敏感，论文中需与均值显著性一并解释。
