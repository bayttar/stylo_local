from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from pipeline_common import GSECT, clean_text, is_missing, load_metadata


CANON = [
    (
        "FRAMEWORK",
        [
            r"\btheory\b",
            r"\btheoretical\b",
            r"\bframework\b",
            r"\bliterature review\b",
            r"\bcritical context\b",
            r"\bscholarship\b",
            r"\bcriticism\b",
            r"\bhistoriography\b",
            r"\bmethodology\b",
            r"\bmethod\b",
            r"\bapproach\b",
            r"\bliterature\b",
        ],
    ),
    ("INTRO", [r"\bintroduction\b", r"\bbackground\b", r"\bcontext\b", r"\bpreamble\b", r"\boverview\b", r"\bpreface\b", r"\bopening\b"]),
    ("DISCUSSION", [r"\bdiscussion\b", r"\bimplications\b", r"\binterpretation\b", r"\bsignificance\b", r"\bbroader context\b", r"\bwider implications\b"]),
    ("CONCLUSION", [r"\bconclusion\b", r"\bconcluding\b", r"\bfinal remarks\b", r"\bcoda\b", r"\bafterword\b", r"\bepilogue\b", r"\bclosing\b", r"\bsummary\b"]),
]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CANONICAL_ORDER = ["INTRO", "FRAMEWORK", "ARGUMENT", "DISCUSSION", "CONCLUSION"]


def canonise(name: str) -> str:
    s = clean_text(name).lower()
    for label, pats in CANON:
        for pattern in pats:
            if re.search(pattern, s):
                return label
    return "ARGUMENT"


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def main() -> None:
    sections_path = GSECT / "sections.jsonl"
    if not sections_path.exists():
        raise FileNotFoundError(f"Missing: {sections_path}")

    meta = load_metadata()
    journal_map = meta.set_index("file_stem")["journal_label"].to_dict()

    rows: list[dict[str, object]] = []
    with sections_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tei_file = clean_text(record.get("tei_file", ""))
            file_stem = tei_file.replace(".tei.xml", "") if tei_file else clean_text(record.get("file_stem", ""))
            journal_label = clean_text(journal_map.get(file_stem, ""))
            if is_missing(journal_label):
                continue

            section_rows: list[dict[str, object]] = []
            abstract = clean_text(record.get("abstract", ""))
            if abstract:
                section_rows.append(
                    {
                        "file_stem": file_stem,
                        "journal_label": journal_label,
                        "section_raw": "Abstract",
                        "section_canon": "INTRO",
                        "word_count": _word_count(abstract),
                    }
                )

            for sec in record.get("sections", []):
                raw_name = clean_text(sec.get("raw_head") or sec.get("norm_head") or "")
                text = clean_text(sec.get("text", ""))
                if not text:
                    continue
                section_rows.append(
                    {
                        "file_stem": file_stem,
                        "journal_label": journal_label,
                        "section_raw": raw_name or "Untitled",
                        "section_canon": canonise(raw_name or ""),
                        "word_count": _word_count(text),
                    }
                )

            total_words = sum(int(row["word_count"]) for row in section_rows)
            if total_words == 0:
                continue
            for row in section_rows:
                row["word_share"] = float(row["word_count"]) / float(total_words)
                rows.append(row)

    if not rows:
        raise RuntimeError("No section rows available for canonical template fingerprinting.")

    long = pd.DataFrame(rows)
    tpl = (
        long.groupby(["journal_label", "section_canon"])["word_share"]
        .mean()
        .reset_index()
        .pivot(index="journal_label", columns="section_canon", values="word_share")
        .fillna(0.0)
        .reset_index()
    )
    for col in CANONICAL_ORDER:
        if col not in tpl.columns:
            tpl[col] = 0.0
    tpl = tpl[["journal_label"] + CANONICAL_ORDER]

    out1 = GSECT / "journal_section_template_canonical.csv"
    tpl.to_csv(out1, index=False)

    freq = (
        long.groupby("section_raw")["file_stem"]
        .count()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"file_stem": "count"})
    )
    out2 = GSECT / "section_raw_name_counts.csv"
    freq.to_csv(out2, index=False)

    print("Saved:")
    print("-", out1)
    print("-", out2)
    print("Unique journals:", tpl["journal_label"].nunique())


if __name__ == "__main__":
    main()
