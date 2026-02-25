from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def main():
    master = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections" / "MASTER_v2.csv"
    df = pd.read_csv(master)

    # remove identifiers and non-stylometric columns
    drop_cols = [
        "file", "file_stem", "title", "doi",
        "publisher", "publisher_crossref",
        "journal", "journal_crossref",
        "journal_label", "authors"
    ]
    drop_cols += [c for c in df.columns if c.endswith("_word_share")]

    Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    Xdf = Xdf.select_dtypes(include=[np.number]).copy()

    # cast to float explicitly
    Xdf = Xdf.astype(float)

    # remove constant columns
    Xdf = Xdf.loc[:, Xdf.std() > 0]

    # impute NaNs with column mean
    imputer = SimpleImputer(strategy="mean")
    Xdf_imputed = pd.DataFrame(
        imputer.fit_transform(Xdf),
        columns=Xdf.columns
    )

    journals = df["journal_label"].fillna("UNKNOWN").astype(str)

    # journal de-meaning
    X_res = Xdf_imputed.copy()
    for j in journals.unique():
        idx = (journals == j)
        mu = Xdf_imputed.loc[idx].mean()
        X_res.loc[idx] = Xdf_imputed.loc[idx] - mu

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_res)

    # PCA
    pca = PCA(n_components=min(10, X.shape[1]), random_state=42)
    Z = pca.fit_transform(X)

    print("Explained variance ratio:")
    print(np.round(pca.explained_variance_ratio_, 4))

    # clustering
    n = len(df)
    k = min(10, max(3, int(np.sqrt(n))))
    km = KMeans(n_clusters=k, random_state=42, n_init=30)
    clusters = km.fit_predict(Z[:, :5])

    out = master.parent / "author_signature_residual_clusters.csv"
    out_df = df[["file", "authors", "journal_label"]].copy()
    out_df["cluster"] = clusters
    out_df.to_csv(out, index=False)

    print("Saved:", out)
    print("k:", k)

if __name__ == "__main__":
    main()
