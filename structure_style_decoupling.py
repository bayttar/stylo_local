from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    master_path = base / "MASTER_v2.csv"
    df = pd.read_csv(master_path)

    # ----- STRUCTURE VECTOR: section shares -----
    share_cols = [c for c in df.columns if c.endswith("_word_share")]
    if not share_cols:
        raise ValueError("No *_word_share columns found in MASTER_v2.csv")

    S = df[share_cols].astype(float).fillna(0.0)

    S_scaled = StandardScaler().fit_transform(S)
    pca_s = PCA(n_components=min(5, S_scaled.shape[1]), random_state=42)
    S_pc = pca_s.fit_transform(S_scaled)

    # ----- STYLE VECTOR: numeric stylometry excluding shares -----
    drop = share_cols + [
        "file", "file_stem", "title", "doi", "authors",
        "journal_label", "journal_crossref", "journal",
        "publisher", "publisher_crossref"
    ]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).astype(float)

    # Drop all-NaN columns (e.g., year)
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        print("Dropping all-NaN numeric columns:", all_nan)
        X = X.drop(columns=all_nan)

    # Impute remaining NaNs with column mean
    imputer = SimpleImputer(strategy="mean")
    X_imp_arr = imputer.fit_transform(X)
    X_imp = pd.DataFrame(X_imp_arr, columns=X.columns)

    # journal de-mean
    journals = df["journal_label"].fillna("UNKNOWN").astype(str).values
    X_res = X_imp.copy()
    for j in pd.unique(journals):
        idx = (journals == j)
        mu = X_imp.loc[idx].mean()
        X_res.loc[idx] = X_imp.loc[idx] - mu

    X_scaled = StandardScaler().fit_transform(X_res)
    pca_x = PCA(n_components=min(5, X_scaled.shape[1]), random_state=42)
    X_pc = pca_x.fit_transform(X_scaled)

    # ----- CORRELATION: structure PCs vs residual style PCs -----
    corr = np.corrcoef(S_pc.T, X_pc.T)[:S_pc.shape[1], S_pc.shape[1]:]
    corr_df = pd.DataFrame(
        corr,
        index=[f"struct_PC{i+1}" for i in range(S_pc.shape[1])],
        columns=[f"style_resid_PC{i+1}" for i in range(X_pc.shape[1])]
    )

    out = base / "structure_style_pc_correlations.csv"
    corr_df.to_csv(out, index=True)

    print("Saved:", out)
    print("Correlation matrix (structure vs residual style PCs):")
    print(corr_df.round(3))

if __name__ == "__main__":
    main()
