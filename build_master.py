from pathlib import Path
import pandas as pd

def main():
    base = Path.home() / "stylo_local" / "stylo_out"
    sections_dir = base / "grobid_sections"

    metrics_csv = base / "per_article_metrics.csv"
    sections_csv = sections_dir / "per_article_section_metrics_wide.csv"
    meta_csv = sections_dir / "metadata_from_tei.csv"

    for p in [metrics_csv, sections_csv, meta_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    df_m = pd.read_csv(metrics_csv)
    df_s = pd.read_csv(sections_csv)
    df_meta = pd.read_csv(meta_csv)

    # join keys:
    # metrics: file = PDF filename, we derive file_stem by stripping .pdf
    df_m["file_stem"] = df_m["file"].astype(str).str.replace(r"\.pdf$", "", regex=True)

    master = df_m.merge(df_s, on="file_stem", how="left").merge(df_meta, on="file_stem", how="left")

    out = sections_dir / "MASTER_metrics_sections_metadata.csv"
    master.to_csv(out, index=False)

    print("Saved:", out)
    print("Rows:", len(master))
    print("Columns:", len(master.columns))

if __name__ == "__main__":
    main()
