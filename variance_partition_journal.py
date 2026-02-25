from pathlib import Path
import pandas as pd
import numpy as np

def eta_squared_by_group(x, g):
    # one-way ANOVA effect size: eta^2 = SS_between / SS_total
    x = np.asarray(x, dtype=float)
    g = np.asarray(g)
    m = np.nanmean(x)
    ss_total = np.nansum((x - m) ** 2)

    ss_between = 0.0
    for lab in pd.unique(g):
        idx = (g == lab)
        if idx.sum() < 2:
            continue
        mg = np.nanmean(x[idx])
        ss_between += idx.sum() * (mg - m) ** 2

    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    master = base / "MASTER_v2.csv"
    df = pd.read_csv(master)

    journals = df["journal_label"].fillna("UNKNOWN").astype(str)

    # numeric stylometry only (exclude section shares)
    drop = [c for c in df.columns if c.endswith("_word_share")]
    drop += ["file","file_stem","title","doi","authors","journal_label","journal_crossref","journal","publisher","publisher_crossref"]
    Xdf = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    Xdf = Xdf.select_dtypes(include=[np.number]).astype(float)

    # simple mean impute
    Xdf = Xdf.fillna(Xdf.mean(numeric_only=True))

    rows = []
    for col in Xdf.columns:
        eta2 = eta_squared_by_group(Xdf[col].values, journals.values)
        rows.append({"feature": col, "eta_squared_journal": eta2})

    out = base / "journal_effect_sizes_eta2.csv"
    res = pd.DataFrame(rows).sort_values("eta_squared_journal", ascending=False)
    res.to_csv(out, index=False)

    print("Saved:", out)
    print()
    print("Top journal-driven features (highest eta^2):")
    print(res.head(25))

if __name__ == "__main__":
    main()
