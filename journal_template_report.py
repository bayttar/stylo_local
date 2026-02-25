from pathlib import Path
import pandas as pd
import numpy as np

def entropy(proportions):
    p = np.array([x for x in proportions if pd.notna(x) and x > 0], dtype=float)
    if p.size == 0:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())

def main():
    master_path = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections" / "MASTER_metrics_sections_metadata.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing: {master_path}. Run build_master.py first.")

    df = pd.read_csv(master_path)

    # section word-share columns end with _word_share (proportion already)
    share_cols = [c for c in df.columns if c.endswith("_word_share") and c.startswith("sec_")]

    # per-article section entropy (template complexity)
    df["section_entropy"] = df[share_cols].apply(lambda r: entropy(r.values), axis=1)

    # journal label: use journal field if present else publisher else UNKNOWN
    df["journal_label"] = df["journal"].fillna("").astype(str).str.strip()
    df.loc[df["journal_label"] == "", "journal_label"] = df["publisher"].fillna("").astype(str).str.strip()
    df.loc[df["journal_label"] == "", "journal_label"] = "UNKNOWN"

    # journal-level mean of section shares + mean entropy
    agg = df.groupby("journal_label")[share_cols + ["section_entropy"]].mean(numeric_only=True).reset_index()

    out_dir = master_path.parent
    out_csv = out_dir / "journal_template_means.csv"
    agg.to_csv(out_csv, index=False)

    out_articles = out_dir / "per_article_template_complexity.csv"
    df[["file", "file_stem", "journal_label", "section_entropy"]].to_csv(out_articles, index=False)

    print("Saved:")
    print("-", out_csv)
    print("-", out_articles)
    print("Journals:", agg.shape[0])

if __name__ == "__main__":
    main()
