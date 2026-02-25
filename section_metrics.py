import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from lexicalrichness import LexicalRichness

SPACY_MODEL = "en_core_web_sm"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SEMICOLON_RE = re.compile(r";")
COLON_RE = re.compile(r":")
DASH_RE = re.compile(r"--")  # keep ASCII-only
SENT_BINS = [
    ("lt12", 0, 11),
    ("12_20", 12, 20),
    ("21_30", 21, 30),
    ("31_40", 31, 40),
    ("41_50", 41, 50),
    ("gt50", 51, 10_000),
]

def safe_key(s: str) -> str:
    s = (s or "Untitled").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Untitled"

def words(text: str) -> List[str]:
    return WORD_RE.findall(text or "")

def bin_counts(sent_lens: List[int]) -> Dict[str, int]:
    c = Counter()
    for wc in sent_lens:
        for label, lo, hi in SENT_BINS:
            if lo <= wc <= hi:
                c[label] += 1
                break
    return dict(c)

def metrics_for_text(nlp, text: str) -> Dict[str, Any]:
    wl = words(text)
    doc = nlp(text)

    sent_lens = [sum(1 for t in s if t.is_alpha) for s in doc.sents]
    total_sents = len(sent_lens) or 1

    mtld = 0.0
    if wl:
        lr = LexicalRichness(" ".join([w.lower() for w in wl]))
        mtld = float(lr.mtld(0.72))

    return {
        "total_words": len(wl),
        "total_sentences": len(sent_lens),
        "avg_sentence_len": float(np.mean(sent_lens)) if sent_lens else 0.0,
        "median_sentence_len": float(np.median(sent_lens)) if sent_lens else 0.0,
        "sd_sentence_len": float(np.std(sent_lens, ddof=1)) if len(sent_lens) > 1 else 0.0,
        "sent_gt_40_count": sum(1 for x in sent_lens if x > 40),
        "sent_lt_12_count": sum(1 for x in sent_lens if x < 12),
        "sent_gt_40_pct": sum(1 for x in sent_lens if x > 40) / total_sents,
        "sent_lt_12_pct": sum(1 for x in sent_lens if x < 12) / total_sents,
        "sentence_bin_counts": bin_counts(sent_lens),
        "semicolons_total": len(SEMICOLON_RE.findall(text or "")),
        "colons_total": len(COLON_RE.findall(text or "")),
        "double_dashes_total": len(DASH_RE.findall(text or "")),
        "mtld": mtld,
    }

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    sections_jsonl = base / "sections.jsonl"
    if not sections_jsonl.exists():
        raise FileNotFoundError(f"Missing: {sections_jsonl}. Run tei_sections_batch.py first.")

    nlp = spacy.load(SPACY_MODEL)
    print("spaCy loaded:", SPACY_MODEL)

    records = []
    with open(sections_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("sections.jsonl is empty.")

    rows = []
    section_freq = Counter()

    for rec in tqdm(records, desc="Per-article section metrics"):
        tei_file = rec.get("tei_file", "")
        file_stem = tei_file.replace(".tei.xml", "") if tei_file else "UNKNOWN"
        row: Dict[str, Any] = {"file_stem": file_stem}

        # Behaviour change (explicit): aggregate multiple divs with same norm_head by concatenation
        # Justification: avoids duplicate columns and matches journal-template analysis better.
        buckets = defaultdict(list)

        abs_text = (rec.get("abstract") or "").strip()
        if abs_text:
            buckets["Abstract"].append(abs_text)

        for sec in rec.get("sections", []):
            name = (sec.get("norm_head") or "Untitled").strip()
            txt = (sec.get("text") or "").strip()
            if not txt:
                continue
            if name == "References":
                continue
            buckets[name].append(txt)

        for sec_name, parts in buckets.items():
            full = "\n\n".join(parts).strip()
            if not full:
                continue

            section_freq[sec_name] += 1

            m = metrics_for_text(nlp, full)
            sk = safe_key(sec_name)
            for k, v in m.items():
                if isinstance(v, dict):
                    row[f"sec_{sk}_{k}"] = json.dumps(v, ensure_ascii=False)
                else:
                    row[f"sec_{sk}_{k}"] = v

            # also store share-of-words for template inference
            row[f"sec_{sk}_word_share"] = m["total_words"]

        # convert word shares to proportions
        share_cols = [c for c in row.keys() if c.endswith("_word_share")]
        total_words_all_secs = sum(float(row[c]) for c in share_cols) if share_cols else 0.0
        if total_words_all_secs > 0:
            for c in share_cols:
                row[c] = float(row[c]) / total_words_all_secs

        rows.append(row)

    df = pd.DataFrame(rows)
    out_wide = base / "per_article_section_metrics_wide.csv"
    df.to_csv(out_wide, index=False)

    freq_df = pd.DataFrame([{"section": k, "count": v} for k, v in section_freq.most_common()])
    out_freq = base / "section_name_frequencies.csv"
    freq_df.to_csv(out_freq, index=False)

    print("Saved:")
    print("-", out_wide)
    print("-", out_freq)

if __name__ == "__main__":
    main()
