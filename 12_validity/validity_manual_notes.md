# 效度检验数据收集 — 人工核验与干预记录

> 生成日期：2026-08-20
> 数据文件：`validity_fundamentals_final.csv`（149 行 = 149 个 MDA 文档，31 家公司）
> 完整证据（解析命中行、页码、来源方法）：`validity_fundamentals_raw.csv`
> 提取缓存：`pdftotext_raw/`（pdftotext -raw，240 个 PDF）

## 一、覆盖情况

| 变量 | 覆盖 | 说明 |
|------|------|------|
| auditor_firm + big4 | 149/149 | Big4 = 62 行（41.6%） |
| total_assets_reported + unit | 149/149 | 单位：RM'000 ×76、RM ×63、RMB'000 ×10 |
| profit_loss_reported + loss_flag | 149/149 | Loss = 35 行（23.5%） |

## 二、人工核验修正（FIXUPS，代码内 hardcode）

| 文档 | 字段 | 值 | 证据 |
|------|------|-----|------|
| MDA_2836_CARLSBG_2021 | profit | 204,365 | 自有 SoPL（part3 p6/pdftotext p120），PyMuPDF 直读 |
| MDA_2836_CARLSBG_2020 | profit | 166,185 | 2021 SoPL 比较列；2020 年报 FS 不在本地文件（含于五年摘要：167.4 RM mil，取报表口径） |
| MDA_2836_CARLSBG_2020 | auditor | PwC | 2020 FS 不在本地；2021 年报审计意见覆盖 2020 比较数，PwC 2021-2024 连续 |
| MDA_3689_FN_2021 | profit | 395,130 | FN 2022 SoPL 比较列（"382,269 395,130"） |
| MDA_5187_HBGLOB_2020 | TA | 419,760 | 自有 SoFP（p57，PyMuPDF），RMB'000 |
| MDA_5187_HBGLOB_2021 | profit | -45,290 | 自有 SoPL 页字体损坏，300DPI 渲染+Windows OCR 读取 |
| MDA_6203_KHEESAN_2022 | profit | -13,735,839 | 自有 SoPL（p51 "Loss for the year 26 (13,735,839)..."）；解析器曾误取 SoCE 行 (58,325,818) |
| MDA_6203_KHEESAN_2024 | TA | 71,898,278 | 自有 SoFP（p59，PyMuPDF），RM 单位 |
| MDA_5008_HARISON_2021 | auditor | PwC | AUDITORS 标签回退抓到了决议行原文，行内含 PricewaterhouseCoopers |

## 三、跨年比较列填补（ta/profit_from_next_year_comparative）

当年报 PDF 内报表缺失/损坏时，用**下一年年报的比较列**填上一年数值：
MDA_2658_AJI_2020、MDA_2658_AJI_2023、MDA_2836_CARLSBG_2020、MDA_4065_PPB_2021、
MDA_5008_HARISON_2021（profit）、MDA_5187_HBGLOB_2021（TA）、MDA_5202_MSM_2021、
MDA_5306_FFB_2023、MDA_7243_MAGMA_2022（profit）。

## 四、外部来源（ta/profit_external_web）

**NESTLE (4707) 2020-2024**：审计财报是独立文档，不在 PDF 文件夹。数值取自财经聚合站
（Investing.com / TipRanks / Bernama / stockanalysis.com），RM'000：
- 2020 TA 2,861,400 / profit 552,710；2021 TA 2,984,830 / 569,810
- 2022 TA 3,554,010 / 620,330；2023 TA 3,569,220 / 659,870；2024 TA 3,649,740 / 415,620

**CNOUHUA (5188) 2020-2024**：报表表格为低分辨率嵌入图片，OCR 不可靠。数值取自
MarketScreener / Investing.com / Yahoo（近似值，RMB'000）：
- 2020 TA ≈196,000 / loss ≈15,940；2021 ≈194,790 / ≈4,080；2022 ≈168,530 / ≈24,690
- 2023 ≈163,060 / ≈7,070；2024 ≈111,240 / ≈46,670

## 五、单位说明与覆盖修正

- AJI (2658) 全年份 RM 实际单位（非 RM'000）；APOLLO (6432) 2024 同。
- HBGLOB (5187) 全年份 RMB'000（2022 年报表六列含两年比较数，与 2020/2021 原报表数字吻合，确认币种一致；其字体损坏导致单位检测误报 RM'000）。
- FFB (5306) 2023 年报以 RM 列报、2024 年报以 RM'000 列报（数值一致：49,934,274 RM = 49,934 RM'000），各自保留原报表单位。
- CNOUHUA 以 RMB 列报。

## 六、已知异常（经核实为真实情况，论文中可酌情披露）

1. 审计师变更：AJI EY→KPMG(2022)、APOLLO BDO→KPMG(2024)、LOTUS Grant Thornton→TGS TW(2023)、HBGLOB UHY↔Folks DFK、KHEESAN 全部 Kreston John & Gan（非四大）。
2. CARLSBG 2022 TA +70%：Carlsberg 新加坡业务同一控制合并导致 2020/2021 比较数重述（2022 年报 5 列 SoFP）；本表各年用原年报口径。
3. HBGLOB 2022/2023 比较数重述（2022 亏损原报 2,302，2023 年报重述为 167,698）——本表用原年报口径。
4. AJI 2024 利润 27.5M→401.4M：出售 Kuchai Lama 土地收益（真实）。
5. MFLOUR 2023→2024 利润 5,285→77,223：家禽周期（真实）。
6. 3A (0012) 2023 亏损：真实（公司实际亏损年）。
7. KHEESAN 2024 集团列多为 "-"（业务已出售，现金壳公司），Loss 取公司层盈利 1,725,976 → loss_flag=0。

## 七、复现方式

```bash
cd ver2
python code/scripts/collect_validity_extract_text.py --root .     # pdftotext -raw 提取（缓存于 12_validity/pdftotext_raw/）
python code/scripts/collect_validity_parse_fundamentals.py --root .   # 解析+修复+跨年填补+外部数据
python code/scripts/collect_validity_build_final.py --root .      # 生成最终 CSV
```
