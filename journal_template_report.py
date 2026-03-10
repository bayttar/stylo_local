from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import GSECT, clean_text, load_metadata


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def entropy(proportions: list[float]) -> float:
    p = np.array([x for x in proportions if pd.notna(x) and x > 0], dtype=float)
    if p.size == 0:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def main() -> None:
    sections_path = GSECT / "sections.jsonl"
    if not sections_path.exists():
        raise FileNotFoundError(f"Missing: {sections_path}. Run tei_sections_batch.py first.")

    meta = load_metadata()
    meta_small = meta[["file_stem", "journal_label"]].drop_duplicates(subset=["file_stem"])
    journal_map = meta_small.set_index("file_stem")["journal_label"].to_dict()

    rows: list[dict[str, object]] = []
    with sections_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tei_file = clean_text(record.get("tei_file", ""))
            file_stem = tei_file.replace(".tei.xml", "") if tei_file else clean_text(record.get("file_stem", ""))
            counts: list[int] = []

            abstract = clean_text(record.get("abstract", ""))
            if abstract:
                counts.append(_word_count(abstract))

            for sec in record.get("sections", []):
                text = clean_text(sec.get("text", ""))
                if text:
                    counts.append(_word_count(text))

            total_words = sum(counts)
            shares = [count / total_words for count in counts if total_words > 0]
            rows.append(
                {
                    "file_stem": file_stem,
                    "journal_label": clean_text(journal_map.get(file_stem, "")),
                    "section_count": len([c for c in counts if c > 0]),
                    "section_entropy": entropy(shares),
                }
            )

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("journal_label", as_index=False)[["section_count", "section_entropy"]]
        .mean(numeric_only=True)
        .rename(columns={"section_count": "canonical_section_count"})
    )

    out_csv = GSECT / "journal_template_means.csv"
    agg.to_csv(out_csv, index=False)

    out_articles = GSECT / "per_article_template_complexity.csv"
    df.to_csv(out_articles, index=False)

    print("Saved:")
    print("-", out_csv)
    print("-", out_articles)
    print("Journals:", agg.shape[0])


if __name__ == "__main__":
    main()
