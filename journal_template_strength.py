from pathlib import Path
import pandas as pd
import numpy as np

def shannon_entropy(p):
    p = np.array(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    tpl_path = base / "journal_section_template_canonical.csv"
    if not tpl_path.exists():
        raise FileNotFoundError(f"Missing: {tpl_path} (run section_template_fingerprint.py first)")

    tpl = pd.read_csv(tpl_path)

    # entropy per journal
    sec_cols = [c for c in tpl.columns if c != "journal_label"]
    ent = []
    dom = []
    for _, r in tpl.iterrows():
        vals = [r[c] for c in sec_cols]
        ent.append(shannon_entropy(vals))
        dom.append(max(vals) if len(vals) else 0.0)

    tpl["section_entropy_bits"] = ent
    tpl["dominant_section_share"] = dom

    out = base / "journal_template_strength.csv"
    tpl.sort_values(["section_entropy_bits"], ascending=True).to_csv(out, index=False)

    print("Saved:", out)
    print()
    print("Lowest entropy (most rigid templates):")
    print(tpl[["journal_label","section_entropy_bits","dominant_section_share"]].sort_values("section_entropy_bits").head(10))
    print()
    print("Highest entropy (most flexible templates):")
    print(tpl[["journal_label","section_entropy_bits","dominant_section_share"]].sort_values("section_entropy_bits", ascending=False).head(10))

if __name__ == "__main__":
    main()
