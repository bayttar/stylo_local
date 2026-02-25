from pathlib import Path
import pandas as pd

def main():
    base = Path.home() / "stylo_local" / "stylo_out"
    sec = base / "grobid_sections"

    metrics_csv = base / "per_article_metrics.csv"
    sections_csv = sec / "per_article_section_metrics_wide.csv"
    meta_enriched = sec / "metadata_enriched.csv"

    for p in [metrics_csv, sections_csv, meta_enriched]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    df_m = pd.read_csv(metrics_csv)
    df_s = pd.read_csv(sections_csv)
    df_meta = pd.read_csv(meta_enriched)

    df_m["file_stem"] = df_m["file"].astype(str).str.replace(r"\.pdf$", "", regex=True)

    master = (
        df_m.merge(df_s, on="file_stem", how="left")
            .merge(df_meta, on="file_stem", how="left")
    )

    # choose journal label
    master["journal_label"] = master.get("journal_crossref", "").fillna("").astype(str).str.strip()
    master.loc[master["journal_label"] == "", "journal_label"] = "UNKNOWN"

    out = sec / "MASTER_v2.csv"
    master.to_csv(out, index=False)
    print("Saved:", out)
    print("Rows:", len(master))
    print("Unique journals:", master["journal_label"].nunique())

if __name__ == "__main__":
    main()
