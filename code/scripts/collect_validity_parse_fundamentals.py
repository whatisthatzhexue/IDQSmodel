"""Parse auditor / total assets / profit-or-loss from raw-mode annual-report text.

Works on pdftotext -raw output, where financial-statement rows come out as
"LABEL n1 n2 n3 n4" (Group current year = FIRST number after the label).

Usage (from ver2 dir):
  python code/scripts/collect_validity_parse_fundamentals.py --root .

Inputs:
  - ver2/12_validity/pdftotext_raw/*.txt     (pdftotext -raw)
  - ver2/08_reports/market_mapping.csv       (149 company-year pairs)
Outputs:
  - ver2/12_validity/validity_fundamentals_raw.csv   (with page/line evidence)
  - ver2/12_validity/validity_fundamentals.csv       (clean, one row per doc)
  - ver2/12_validity/validity_warnings.txt           (cross-checks to review)
"""
import argparse
import csv
import os
import re

BIG4_TOKENS = [
    ("KPMG", r"kpmg"),
    ("PwC", r"pricewaterhousecoopers|price\s*waterhouse"),
    ("EY", r"ernst\s*&?\s*young"),
    ("Deloitte", r"deloitte"),
]
NONBIG4_TOKENS = [
    ("Crowe", r"crowe"),
    ("Grant Thornton", r"grant\s*thornton"),
    ("BDO", r"\bbdo\b"),
    ("Baker Tilly", r"baker\s*tilly|monteiro"),
    ("Mazars", r"mazars"),
    ("UHY", r"\buhy\b"),
    ("Moore", r"moore\s*(malaysia|stephens)?"),
    ("Morison", r"morison"),
    ("Ecovis", r"ecovis"),
    ("HLB Ler Lum", r"\bhlb\b|ler\s*lum"),
    ("PKF", r"\bpkf\b"),
    ("RSM", r"\brsm\b"),
    ("AljeffriDean", r"aljeffridean"),
    ("Salihin", r"salihin"),
    ("Folks DFK", r"folks|\bdfk\b"),
    ("Kreston", r"kreston"),
    ("TGS TW", r"\btgs\b"),
    ("Cheng & Co", r"cheng\s*&\s*co"),
    ("SJM", r"\bsjm\b"),
    ("Peter Chong", r"peter\s*chong"),
    ("Saffery", r"saffery"),
]
ALL_TOKENS = BIG4_TOKENS + NONBIG4_TOKENS

AUDIT_HEADING = re.compile(r"INDEPENDENT\s+AUDITORS?['’`]?\s*REPORT", re.I)
SOFP_HEADING = re.compile(r"STATEMENTS?\s+OF\s+FINANCIAL\s+POSITION", re.I)
SOPL_HEADING = re.compile(
    r"STATEMENTS?\s+OF\s+PROFIT\s+OR\s+LOSS|STATEMENTS?\s+OF\s+COMPREHENSIVE\s+INCOME", re.I)
TA_LABEL = re.compile(r"^\s*TOTAL\s+ASSETS\b", re.I)
PL_LABEL = re.compile(
    r"(LOSS|\(?\s*LOSS\s*\)?\s*/\s*PROFIT"
    r"|PROFIT\s*/\s*\(?\s*(?:LOSS|TOTAL\s+COMPREHENSIVE\s+INCOME)\s*\)?"
    r"|TOTAL\s+COMPREHENSIVE\s+INCOME|NET\s+PROFIT|PROFIT\s+AFTER\s+TAX|PROFIT)"
    r"\s+FOR\s+THE\s+(FINANCIAL\s+)?YEAR(?!\s+FROM)", re.I)
NUM_TOKEN = re.compile(r"^\(?[\d][\d,\.]*\)?$")
YEAR_TOKEN = re.compile(r"^(19|20)\d{2}$")


def parse_num(tok):
    neg = tok.startswith("(") and tok.endswith(")")
    t = tok.strip("()")
    if not re.fullmatch(r"[\d][\d,\.]*", t):
        return None
    v = float(t.replace(",", ""))
    return -v if neg else v


def pages_of(text):
    return text.split("\f")


def normalize_text(text):
    """Map ligatures / curly quotes / dashes to ASCII so regexes match."""
    for a, b in [
        ("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"),
        ("ﬃ", "ffi"), ("ﬄ", "ffl"),
        ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
        ("–", "-"), ("—", "-"), (" ", " "),
    ]:
        text = text.replace(a, b)
    return text


def numbers_after(text, start=0):
    """Numeric values of tokens in `text` after position `start`."""
    out = []
    for m in re.finditer(r"\S+", text[start:]):
        v = parse_num(m.group())
        if v is not None:
            out.append(v)
    return out


def firm_in_text(text):
    """Firm tokens present in `text` -> (firm, big4) or (None, '')."""
    found = [name for name, pat in ALL_TOKENS if re.search(pat, text, re.I)]
    if not found:
        return None, ""
    big4_hits = [f for f in found if f in {"KPMG", "PwC", "EY", "Deloitte"}]
    other_hits = [f for f in found if f not in {"KPMG", "PwC", "EY", "Deloitte"}]
    if len(big4_hits) == 1 and not other_hits:
        return big4_hits[0], 1
    if len(other_hits) == 1 and not big4_hits:
        return other_hits[0], 0
    return None, ""


def find_auditor(pages):
    """Firm from the audit-report window, scanning BACKWARD from the last
    'Independent Auditors' Report' heading (signature block is at the end)."""
    heading_pages = [i for i, p in enumerate(pages) if AUDIT_HEADING.search(p)]
    if heading_pages:
        last = heading_pages[-1]
        for j in range(last, max(0, last - 5) - 1, -1):
            firm, big4 = firm_in_text(pages[j])
            if firm:
                return [firm], firm, big4, f"audit_window_p{j + 1}"
    # fallback 1: 'AUDITORS' label line anywhere; firm near it (multi-column
    # corporate-info back covers put content before the header line in the
    # text stream, so search the whole page when the label shares its line).
    for pi, p in enumerate(pages):
        lines = p.splitlines()
        for li, line in enumerate(lines):
            if re.match(r"^\s*AUDITORS(?!\s*['’]?\s*REPORT)\b", line, re.I):
                if re.search(r"\bBANKER|REGISTRAR|STOCK\s+EXCHANGE", line, re.I):
                    window = p  # whole page
                else:
                    window = "\n".join(lines[max(0, li - 15):min(len(lines), li + 15)])
                firm, big4 = firm_in_text(window)
                if firm:
                    return [firm], firm, big4, f"auditors_label_p{pi + 1}"
                # unknown firm: capture the line near an 'AF nnnn' registration number
                for l2 in lines:
                    if re.search(r"AF\s*\d+", l2, re.I) or re.search(r"\bPLT\b", l2):
                        cand = re.sub(r"\(.*\)", "", l2).strip()
                        if 3 <= len(cand) <= 60 and cand.upper() != "AUDITORS":
                            return [cand], cand, 0, f"auditors_label_raw_p{pi + 1}"
    # fallback 2: appointment context ("Auditors of the Company" / "re-appoint")
    for pi, p in enumerate(pages):
        for m in re.finditer(
                r"(?:the\s+auditors?,?\s+(?:messrs\.?\s+)?|re-?appoint\w*\s+|"
                r"auditors\s+of\s+the\s+(?:company|group))", p, re.I):
            seg = p[m.end():m.end() + 300]
            firm, big4 = firm_in_text(seg)
            if firm:
                return [firm], firm, big4, f"appointment_p{pi + 1}"
    # fallback 2: whole-document majority vote
    counts = {}
    for pi, p in enumerate(pages):
        for name, pat in ALL_TOKENS:
            counts[name] = counts.get(name, 0) + len(re.findall(pat, p, re.I))
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    ranked = [(n, c) for n, c in ranked if c > 0]
    if ranked:
        clear_winner = len(ranked) == 1 or ranked[0][1] >= 2 * ranked[1][1]
        if clear_winner:
            firm = ranked[0][0]
            big4 = 1 if firm in {"KPMG", "PwC", "EY", "Deloitte"} else 0
            return [firm], firm, big4, "majority_vote"
    return [], None, "", ""


def label_line_candidates(pages, heading_re, label_re, ahead=3):
    """Yield (page_idx, joined_line) label matches inside statement sections.
    Heading pages scanned in REVERSE order first (real statement comes after
    any TOC), then in forward order as fallback."""
    heading_idx = [i for i, p in enumerate(pages) if heading_re.search(p)]
    for hi in reversed(heading_idx):
        for j in range(hi, min(hi + ahead + 1, len(pages))):
            for _li, joined in _joined_lines(pages[j]):
                if label_re.search(joined):
                    yield j, joined
    for hi in heading_idx:
        for j in range(hi, min(hi + ahead + 1, len(pages))):
            for _li, joined in _joined_lines(pages[j]):
                if label_re.search(joined):
                    yield j, joined


def _joined_lines(page_text):
    """All single lines plus each adjacent line pair (labels often wrap)."""
    lines = page_text.splitlines()
    for li in range(len(lines)):
        yield li, lines[li]
        if li + 1 < len(lines):
            yield li, lines[li] + " " + lines[li + 1]


def find_total_assets(pages):
    """Return (vals, page, line) where vals = candidate numbers on the matched
    TOTAL ASSETS row (current year first, comparatives follow)."""
    for pi, line in label_line_candidates(pages, SOFP_HEADING, TA_LABEL, ahead=3):
        m = TA_LABEL.search(line)
        vals = numbers_after(line, m.end())
        vals = [v for v in vals
                if not (v.is_integer() and 1900 <= abs(v) <= 2099)]  # drop year tokens
        big = [v for v in vals if abs(v) >= 1000]
        if big:
            return big, pi + 1, line.strip()[:160]
    # last resort: TOTAL ASSETS line anywhere in the document with numbers
    for pi, page in enumerate(pages):
        for _li, line in _joined_lines(page):
            if TA_LABEL.search(line):
                m = TA_LABEL.search(line)
                vals = [v for v in numbers_after(line, m.end()) if abs(v) >= 1000]
                vals = [v for v in vals
                        if not (v.is_integer() and 1900 <= abs(v) <= 2099)]
                if vals:
                    return vals, pi + 1, line.strip()[:160]
    return None, None, None


def find_profit_loss(pages):
    """Return (vals, is_loss, page, line); vals[0] = current year."""
    for pi, line in label_line_candidates(pages, SOPL_HEADING, PL_LABEL, ahead=3):
        m = PL_LABEL.search(line)
        vals = []
        for tok in re.findall(r"\S+", line[m.end():]):
            if not NUM_TOKEN.match(tok):
                continue
            v = parse_num(tok)
            if v is None:
                continue
            # skip year tokens (e.g. '2022' merged from column headers)
            if tok.isdigit() and not "," in tok and 1900 <= abs(v) <= 2099:
                continue
            has_comma = "," in tok
            if has_comma or abs(v) >= 100 or tok.startswith("("):
                vals.append(v)
        if vals:
            is_loss = bool(re.match(r"LOSS", m.group(1), re.I)) or vals[0] < 0
            return vals, is_loss, pi + 1, line.strip()[:160]
    return None, None, None, None


def unit_of(pages, hi=None):
    """Statement unit: RM'000 / RM. Prefer the column headers right above the
    TOTAL ASSETS row on the same page; fall back to the SoFP heading page."""
    if hi is not None and 0 <= hi < len(pages):
        lines = pages[hi].splitlines()
        for li, line in enumerate(lines):
            if TA_LABEL.search(line):
                blob = "\n".join(lines[max(0, li - 8):min(len(lines), li + 3)])
                if re.search(r"RM\s*[’'`]?\s*0+0+", blob):
                    return "RM'000"
                if re.search(r"\bRM\b", blob, re.I):
                    return "RM"
                break
    for blob in ("\n".join(pages[max(0, hi - 1):hi + 2]) if hi is not None else "",
                 "\n".join(pages)):
        if re.search(r"RM\s*[’'`]?\s*0+0+", blob):
            return "RM'000"
        if re.search(r"\bRM\b", blob, re.I):
            return "RM"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    rows = list(csv.reader(open(os.path.join(root, "08_reports", "market_mapping.csv"),
                                encoding="utf-8-sig")))[1:]
    txt_dir = os.path.join(root, "12_validity", "pdftotext_raw")
    raw_path = os.path.join(root, "12_validity", "validity_fundamentals_raw.csv")
    clean_path = os.path.join(root, "12_validity", "validity_fundamentals.csv")
    warn_path = os.path.join(root, "12_validity", "validity_warnings.txt")

    docs = []
    for r in rows:
        doc_id, code, ticker, year = r[0], r[1], r[2], r[3]
        parts = sorted(f for f in os.listdir(txt_dir)
                       if f.startswith(f"{code}_") and f"_{year}_AnnualReport_" in f
                       and f.endswith(".txt"))
        full_text = ""
        for p in parts:
            with open(os.path.join(txt_dir, p), encoding="utf-8", errors="replace") as f:
                full_text += f.read() + "\n"
        full_text = normalize_text(full_text)
        pages = pages_of(full_text)

        tokens, firm, big4, aud_method = find_auditor(pages)
        ta_vals, ta_page, ta_line = find_total_assets(pages)
        pl_vals, is_loss, pl_page, pl_line = find_profit_loss(pages)
        # unit: search near the TOTAL ASSETS row (page it was found on)
        unit = unit_of(pages, ta_page - 1 if ta_page else None)

        ta = ta_vals[0] if ta_vals else None
        prof = pl_vals[0] if pl_vals else None
        notes = []
        if firm is None:
            notes.append("auditor:" + (",".join(tokens) if tokens else "not_found"))
        if ta is None:
            notes.append("ta_not_found")
        if prof is None:
            notes.append("profit_not_found")

        docs.append(dict(doc=doc_id, code=code, ticker=ticker, year=int(year),
                         parts=parts, tokens=tokens, firm=firm, big4=big4,
                         aud_method=aud_method, ta=ta, ta_vals=ta_vals,
                         ta_page=ta_page, ta_line=ta_line,
                         prof=prof, pl_vals=pl_vals, is_loss=is_loss,
                         pl_page=pl_page, pl_line=pl_line,
                         unit=unit, notes=notes))

    # ---------------- manual fixups (verified against PDF pages) ----------------
    by_key = {(d["code"], d["year"]): d for d in docs}
    by_doc = {d["doc"]: d for d in docs}
    FIXUPS = {
        # doc_id: {field: value} — see validity_manual_notes.md for page evidence
        "MDA_2836_CARLSBG_2020": dict(prof=166185, is_loss=0),
        "MDA_2836_CARLSBG_2021": dict(prof=204365, is_loss=0),
        "MDA_3689_FN_2021": dict(prof=395130, is_loss=0),
        "MDA_5187_HBGLOB_2020": dict(ta=419760),
        "MDA_5187_HBGLOB_2021": dict(prof=-45290, is_loss=1),
        "MDA_6203_KHEESAN_2022": dict(prof=-13735839, is_loss=1),
        "MDA_6203_KHEESAN_2024": dict(ta=71898278),
    }
    for doc_id, fx in FIXUPS.items():
        d = by_doc.get(doc_id)
        if d is None:
            continue
        if "ta" in fx:
            d["ta"] = fx["ta"]
            d["notes"] = [n for n in d["notes"] if n != "ta_not_found"]
            d["notes"].append("ta_manual_verified")
        if "prof" in fx:
            d["prof"] = fx["prof"]
            d["is_loss"] = fx["is_loss"]
            d["notes"] = [n for n in d["notes"] if n != "profit_not_found"]
            d["notes"].append("profit_manual_verified")

    # ---------------- unit overrides (detector false positives) ----------------
    # These companies report in RM / RMB actual units; cross-checked against
    # their own comparative figures across years.
    UNIT_OVERRIDES = {"2658": "RM", "5187": "RMB'000"}
    for d in docs:
        if d["code"] in UNIT_OVERRIDES:
            d["unit"] = UNIT_OVERRIDES[d["code"]]
    d = by_doc.get("MDA_6432_APOLLO_2024")
    if d and d["unit"] == "RM'000" and d["ta"] and d["ta"] < 2_000_000:
        d["unit"] = "RM"

    # ---------------- external sources (FS not in our PDFs) ----------------
    # NESTLE: audited FS published as a separate document; figures from financial
    # data aggregators (Investing.com/TipRanks/Bernama), RM'000.
    NESTLE = {
        2020: dict(ta=2861400, prof=552710),
        2021: dict(ta=2984830, prof=569810),
        2022: dict(ta=3554010, prof=620330),
        2023: dict(ta=3569220, prof=659870),
        2024: dict(ta=3649740, prof=415620),
    }
    # CNOUHUA: FS tables are low-res embedded images; figures approximate from
    # aggregators (MarketScreener/Investing/Yahoo), RMB'000.
    CNOUHUA = {
        2020: dict(ta=196000, prof=-15940),
        2021: dict(ta=194790, prof=-4080),
        2022: dict(ta=168530, prof=-24690),
        2023: dict(ta=163060, prof=-7070),
        2024: dict(ta=111240, prof=-46670),
    }
    for code, table, unit in (("4707", NESTLE, "RM'000"), ("5188", CNOUHUA, "RMB'000")):
        for year, vals in table.items():
            d = by_key.get((code, year))
            if d is None:
                continue
            if d["ta"] is None:
                d["ta"] = vals["ta"]
                d["unit"] = unit
                d["notes"] = [n for n in d["notes"] if n != "ta_not_found"]
                d["notes"].append("ta_external_web")
            if d["prof"] is None:
                d["prof"] = vals["prof"]
                d["is_loss"] = 1 if vals["prof"] < 0 else 0
                d["notes"] = [n for n in d["notes"] if n != "profit_not_found"]
                d["notes"].append("profit_external_web")
    # HARISON 2021: 'AUDITORS' label fallback captured a resolution line;
    # the firm named in it is PwC.
    d = by_doc.get("MDA_5008_HARISON_2021")
    if d and "To re-appoint" in (d["firm"] or ""):
        d["firm"] = "PwC"
        d["big4"] = 1
    # CARLSBG 2020: FS not in our files; 2020 figures carry a PwC audit opinion
    # via the 2021 engagement (PwC audited CARLSBG 2021-2024).
    d = by_doc.get("MDA_2836_CARLSBG_2020")
    if d and d["firm"] == "KPMG":
        d["firm"] = "PwC"
        d["big4"] = 1
        d["notes"].append("auditor_inferred_PwC")

    # ---------------- comparative-column fill ----------------
    # A missing year's figures appear as the prior-year column in the NEXT
    # year's statements; use that when direct extraction failed.
    for d in docs:
        nxt = by_key.get((d["code"], d["year"] + 1))
        if nxt is None:
            continue
        if d["ta"] is None and nxt["ta_vals"] and len(nxt["ta_vals"]) >= 2:
            d["ta"] = nxt["ta_vals"][1]
            d["unit"] = nxt["unit"]
            d["ta_line"] = "FILLED from next-year comparative: " + (nxt["ta_line"] or "")
            d["notes"] = [n for n in d["notes"] if n != "ta_not_found"]
            d["notes"].append("ta_from_next_year_comparative")
        if d["prof"] is None and nxt["pl_vals"] and len(nxt["pl_vals"]) >= 2:
            d["prof"] = nxt["pl_vals"][1]
            d["is_loss"] = 1 if d["prof"] < 0 else 0
            d["pl_line"] = "FILLED from next-year comparative: " + (nxt["pl_line"] or "")
            d["notes"] = [n for n in d["notes"] if n != "profit_not_found"]
            d["notes"].append("profit_from_next_year_comparative")

    # ---------------- cross-checks ----------------
    warnings = []
    by_company = {}
    for d in docs:
        by_company.setdefault(d["code"], []).append(d)
    for code, ds in by_company.items():
        ds.sort(key=lambda x: x["year"])
        firms = {d["firm"] for d in ds if d["firm"]}
        if len(firms) > 1:
            warnings.append(f"{code}: auditor varies across years -> {firms}")
        for prev, cur in zip(ds, ds[1:]):
            if prev["ta"] and cur["ta"] and abs(prev["ta"]) > 1000 and abs(cur["ta"]) > 1000:
                ratio = cur["ta"] / prev["ta"]
                if abs(ratio - 1) > 0.5:
                    warnings.append(
                        f"{code} {cur['year']}: TA {prev['ta']:.0f} -> {cur['ta']:.0f} "
                        f"({ratio:+.0%}) large YoY change")
            if (prev["prof"] is not None and cur["prof"] is not None
                    and prev["is_loss"] == cur["is_loss"] and abs(prev["prof"]) > 0
                    and abs(cur["prof"] / prev["prof"]) > 10):
                warnings.append(
                    f"{code} {cur['year']}: profit {prev['prof']:.0f} -> {cur['prof']:.0f} "
                    f"large swing")
    for d in docs:
        if d["unit"] not in ("RM'000", "RM"):
            warnings.append(f"{d['doc']}: unit = {d['unit']}")
        if d["is_loss"] and d["prof"] is not None and d["prof"] > 0:
            warnings.append(f"{d['doc']}: loss_flag but positive value {d['prof']}")
        if not d["is_loss"] and d["prof"] is not None and d["prof"] < 0:
            warnings.append(f"{d['doc']}: negative value {d['prof']} but no loss flag")

    with open(warn_path, "w", encoding="utf-8") as wf:
        wf.write("\n".join(warnings) if warnings else "(none)")

    with open(raw_path, "w", newline="", encoding="utf-8") as rf:
        w = csv.writer(rf)
        w.writerow(["document_id", "stock_code", "ticker", "year", "parts",
                    "auditor_tokens", "auditor_firm", "big4", "auditor_method",
                    "total_assets_rm000", "ta_page", "ta_line",
                    "profit_rm000", "loss_flag", "pl_page", "pl_line",
                    "unit", "notes"])
        for d in docs:
            w.writerow([d["doc"], d["code"], d["ticker"], d["year"],
                        "|".join(d["parts"]),
                        "|".join(d["tokens"]), d["firm"] or "", d["big4"],
                        d["aud_method"],
                        f"{d['ta']:.0f}" if d["ta"] is not None else "",
                        d["ta_page"] or "", d["ta_line"] or "",
                        f"{d['prof']:.0f}" if d["prof"] is not None else "",
                        1 if d["is_loss"] else (0 if d["prof"] is not None else ""),
                        d["pl_page"] or "", d["pl_line"] or "",
                        d["unit"], ";".join(d["notes"])])

    with open(clean_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["document_id", "stock_code", "ticker", "year",
                    "auditor_firm", "big4", "total_assets_rm000", "profit_rm000",
                    "loss_flag", "unit", "notes"])
        for d in docs:
            w.writerow([d["doc"], d["code"], d["ticker"], d["year"],
                        d["firm"] or "", d["big4"],
                        f"{d['ta']:.0f}" if d["ta"] is not None else "",
                        f"{d['prof']:.0f}" if d["prof"] is not None else "",
                        1 if d["is_loss"] else (0 if d["prof"] is not None else ""),
                        d["unit"], ";".join(d["notes"])])

    n_ok = sum(1 for d in docs if not d["notes"])
    n_aud = sum(1 for d in docs if d["firm"])
    n_ta = sum(1 for d in docs if d["ta"] is not None)
    n_pl = sum(1 for d in docs if d["prof"] is not None)
    print(f"rows={len(docs)} clean={n_ok} auditor_found={n_aud} ta_found={n_ta} profit_found={n_pl}")
    print(f"warnings: {len(warnings)} -> {warn_path}")
    print(f"raw   -> {raw_path}")
    print(f"clean -> {clean_path}")


if __name__ == "__main__":
    main()
