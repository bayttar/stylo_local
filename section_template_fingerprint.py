from pathlib import Path
import pandas as pd
import re

CANON = [
    ("ABSTRACT", [r"\babstract\b"]),
    ("INTRO", [r"\bintroduction\b", r"\bbackground\b", r"\boverview\b"]),
    ("LITREV", [r"\bliterature\b", r"\btheoretical\b", r"\bframework\b", r"\bstate of the art\b"]),
    ("METHODS", [r"\bmethod\b", r"\bmethodology\b", r"\bmaterials\b", r"\bdata\b", r"\bcorpus\b"]),
    ("RESULTS", [r"\bresults\b", r"\bfindings\b", r"\banalysis\b"]),
    ("DISCUSSION", [r"\bdiscussion\b", r"\bimplications\b"]),
    ("CONCLUSION", [r"\bconclusion\b", r"\bconcluding\b", r"\bfinal\b"]),
    ("ACK", [r"\backnowledg", r"\bthanks\b"]),
    ("REFERENCES", [r"\breferences\b", r"\bbibliograph", r"\bworks cited\b"]),
]

def canonise(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    for label, pats in CANON:
        for p in pats:
            if re.search(p, s):
                return label
    return "OTHER"

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    master = base / "MASTER_v2.csv"
    if not master.exists():
        raise FileNotFoundError(f"Missing: {master}")

    df = pd.read_csv(master)

    # columns like sec_<something>_word_share
    share_cols = [c for c in df.columns if c.startswith("sec_") and c.endswith("_word_share")]
    if not share_cols:
        raise ValueError("No section word_share columns found in MASTER_v2.csv")

    # build long table: one row per (paper, section)
    rows = []
    for _, r in df.iterrows():
        for c in share_cols:
            raw_name = c[len("sec_"):-len("_word_share")]
            share = r[c]
            rows.append({
                "file": r.get("file", ""),
                "journal_label": r.get("journal_label", "UNKNOWN"),
                "section_raw": raw_name,
                "section_canon": canonise(raw_name),
                "word_share": share
            })

    long = pd.DataFrame(rows)
    long = long.dropna(subset=["word_share"])

    # journal template: mean share per canonical section
    tpl = (long.groupby(["journal_label", "section_canon"])["word_share"]
              .mean()
              .reset_index()
              .pivot(index="journal_label", columns="section_canon", values="word_share")
              .fillna(0.0)
              .reset_index())

    out1 = base / "journal_section_template_canonical.csv"
    tpl.to_csv(out1, index=False)

    # global distribution of raw section names (for debugging)
    freq = (long.groupby(["section_raw"])["file"].count()
              .sort_values(ascending=False)
              .reset_index()
              .rename(columns={"file": "count"}))
    out2 = base / "section_raw_name_counts.csv"
    freq.to_csv(out2, index=False)

    print("Saved:")
    print("-", out1)
    print("-", out2)
    print("Unique journals:", tpl["journal_label"].nunique())

if __name__ == "__main__":
    main()
