from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pipeline_common import GSECT, merge_metrics_and_metadata, present_style_metrics


def main() -> None:
    df = merge_metrics_and_metadata()
    metric_cols = present_style_metrics(df)
    if not metric_cols:
        raise RuntimeError("No explicit style metrics available for residual clustering.")

    Xdf = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.loc[:, ~Xdf.isna().all()]
    Xdf = Xdf.loc[:, Xdf.std(skipna=True) > 0]

    imputed = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(Xdf), columns=Xdf.columns)
    journals = df["journal_label"].astype(str)

    residual = imputed.copy()
    for journal in journals.unique():
        idx = journals == journal
        residual.loc[idx] = imputed.loc[idx] - imputed.loc[idx].mean()

    Z = PCA(n_components=min(10, residual.shape[1]), random_state=42).fit_transform(
        StandardScaler().fit_transform(residual)
    )

    n = len(df)
    k = min(10, max(3, int(np.sqrt(n))))
    clusters = KMeans(n_clusters=k, random_state=42, n_init=30).fit_predict(Z[:, : min(5, Z.shape[1])])

    out = GSECT / "author_signature_residual_clusters.csv"
    out_df = df[[c for c in ["file", "file_stem", "title", "doi", "journal_label"] if c in df.columns]].copy()
    out_df["cluster"] = clusters
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    print("Saved:", out)
    print("k:", k)


if __name__ == "__main__":
    main()
