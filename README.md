# ver2 — IDQS 披露质量评分流水线（生产版）

基于本地 Ollama（qwen3:8b）对马来西亚 F&B 上市公司 **年报 MD&A** 与**财经新闻**进行五维度披露质量评分，并完成信度（ICC）、抗长度干扰、稳定性与消融等论文所需检验。

> 研究主题：基于 LLM 的信息披露质量评分（IDQS）模型构建与检验
> 论文截止：2026-07-30（原定），数据范围 2020-2024，31 家 Main Market 公司

---

## 目录结构

```
ver2/
├── code/
│   ├── scripts/          # 评分与论文生成脚本（~90 个，核心见下文）
│   └── tests/            # 40+ 个 pytest 测试
├── source/               # 规范输入数据（唯一权威副本）
│   ├── 00_scorebook/     # mda_scorebook.yaml / news_scorebook.yaml（维度+锚点定义）
│   ├── 01_registry/      # mda_registry.csv（149 文档）/ news_registry.csv（125 篇）
│   ├── 02_extracted_text/# mda/（149 txt）、news_clean/（125 txt）
│   ├── 03_numeric_checks/# AR02 数字检查 JSON
│   └── 04_prompts/       # 提示词模板（参考版，运行时由代码生成）
├── 00_scorebook/ 01_registry/ 02_extracted_text/ 03_numeric_checks/ 04_prompts/
│                         # 与 source/ 相同内容，供脚本以 ver2/ 为根直接运行
├── 05_raw_model_outputs/ # 原始 Ollama 输出（可重生成，git 忽略）
├── 06_ratings/           # 聚合评分（可重生成，git 忽略）
├── 08_reports/           # 报告（含人工挑选的 ablation_sample.csv、market_mapping.csv）
├── 11_stability_analysis/# 稳定性分析报告
├── PAPER_OUTPUT/         # 论文就绪输出（表格、图表、各章节）
├── FINAL_OUTPUT/         # 最终冻结数据集
├── paper_output/         # 历史论文输出副本
├── final_deliverable*/   # 三个打包交付版本
└── result/               # 历史完整输出树（体积大，git 忽略）
```

## 环境

- Python：`D:/MySoftware/Anaconda3/python.exe`（3.13），依赖 pandas / numpy / scipy / pyyaml / pytest / matplotlib
- Ollama：`http://127.0.0.1:11434`，模型 `qwen3:8b`
- 评分参数（`code/scripts/scoring_utils.py` 的 `DEFAULT_ENV`）：temperature=0，seed=42，num_ctx=4096，num_predict=192

## 评分维度与权重

| MDA（149 文档） | 权重 | News（125 篇） | 权重 |
|-----------------|------|-----------------|------|
| AR01 内容完整性 | 15% | N01 事实准确性与内部一致性 | 30% |
| AR02 数字一致性与可追溯性 | 25% | N02 来源透明度与可验证性 | 25% |
| AR03 经营分析具体性 | 30% | N03 平衡性与客观性 | 15% |
| AR04 风险与前瞻性披露 | 20% | N04 相关性与信息增量 | 20% |
| AR05 可读性与结构清晰度 | 10% | N05 表达清晰度与标题匹配 | 10% |

LLM 输出每维度整数 1-5；Python 聚合：`std=(raw-1)×25` → 加权总分 0-100 → 等级 A≥85 / B≥70 / C≥55 / D<55。

## 快速开始

```bash
cd ver2

# 0. 环境检查
"D:/MySoftware/Anaconda3/python.exe" code/scripts/check_ollama.py

# 1. 上下文预算烟幕测试（门禁，30 次调用）
"D:/MySoftware/Anaconda3/python.exe" code/scripts/run_context_budget_smoke_test.py --root .

# 2. MDA 全量评分（149 文档 × 5 维度 = 745 次调用，约 40 分钟）
"D:/MySoftware/Anaconda3/python.exe" code/scripts/run_single_dimension_scoring.py \
  --doc-type mda --output-name mda_final_full --ratings-prefix mda_final --root .

# 3. News 全量评分（125 × 5 = 625 次调用）
"D:/MySoftware/Anaconda3/python.exe" code/scripts/run_single_dimension_scoring.py \
  --doc-type news --output-name news_final_full --ratings-prefix news_final --root .

# 4. ICC 重测（MDA 两轮；News 同理，换 doc-type/news 与输出名）
"D:/MySoftware/Anaconda3/python.exe" code/scripts/run_mda_final_repeatability_test.py --root .

# 5. 字数-评分相关性（抗长度捷径检验）
"D:/MySoftware/Anaconda3/python.exe" code/scripts/compute_length_score_correlation.py \
  --root . --mda-run mda_final_full --news-run news_final_full

# 6. 消融实验（4 变体 × 25 样本 × 5 维度）
"D:/MySoftware/Anaconda3/python.exe" code/scripts/run_ablation_variant_scoring.py --root . --tone strict
"D:/MySoftware/Anaconda3/python.exe" code/scripts/run_ablation_experiment.py --root .

# 7. 一键论文就绪（需先完成交叉验证等前置）
"D:/MySoftware/Anaconda3/python.exe" code/scripts/run_paper_ready_pipeline.py --root .
```

## 核心脚本速查

| 脚本 | 作用 |
|------|------|
| `llm_clients.py` | OllamaClient：/api/chat 调用、JSON 修复、健康检查 |
| `scoring_utils.py` | 全局配置（DEFAULT_ENV）、维度定义、CSV 读写 |
| `run_single_dimension_scoring.py` | 逐维度评分引擎（MDA 与 News 通用） |
| `run_mda_final_repeatability_test.py` | MDA ICC 两轮 + 对比报告 |
| `compute_length_score_correlation.py` | 字数-评分 Pearson/Spearman 相关性 |
| `run_ablation_variant_scoring.py` | 消融变体评分（strict/lenient/mild_* 四变体） |
| `run_ablation_experiment.py` | 消融显著性检验（配对 T / 比例检验，scipy） |
| `run_context_budget_smoke_test.py` | 20 轮重复 + 矩阵上下文预算验证 |
| `run_statistical_stability_analysis.py` | 统计稳定性分析 |
| `run_paper_ready_pipeline.py` | 收敛循环 + 数据冻结 + 论文输出生成 |

## 已知结果（2026-09 批次）

- **ICC**：MDA 149/149 完全一致（repeatability_pass_rate=1.0）；News 125/125 完全一致
- **字数相关性**：MDA r=-0.042，News r=-0.172（均 p>0.05，无长度捷径）
- **市场覆盖**：149 个 MDA 文档全部为 Main Market（ACE/LEAP 公司年报无标准 MD&A 章节，未纳入）

## 重要注意事项（踩坑记录）

1. **所有脚本必须在 ver2/ 目录下执行并加 `--root .`**。脚本默认 `SCORING_ROOT` 指向 `ver2/code/`（历史遗留），不加会找不到 registry/文本。
2. **`llm_clients.py` 中 `format` 已改为 `"json"` 字符串**。原 schema dict 会让 qwen3:8b 返回 HTTP 500（"failed to load model vocabulary required for format"）；schema 校验改在 Python 侧完成。
3. **`run_mda_final_repeatability_test.py` 对 run_b 硬编码 `resume=False`**：每次重跑会完全重评 run_b。修复个别失败维度请直接调用 `run_single_dimension_scoring.py`，不要重跑整个 repeatability 脚本。
4. **跨天漂移**：temperature=0 在同会话内完全可复现，但跨天运行可能漂移 1 个维度分（Q4_K_M 量化 + GPU 非确定性）。论文相关批次需在同一会话内连续完成，或注明限制。
5. **删除大目录时优先用 `--overwrite True` 参数**而非 rm，脚本支持原地重建。
6. **消融样本基线**：`08_reports/ablation_sample.csv` 的分数来自挑选时点的基线评分；重跑全量后需重新校验样本类别归属。

## 测试

```bash
cd ver2
"D:/MySoftware/Anaconda3/python.exe" -m pytest code/tests -q
```
