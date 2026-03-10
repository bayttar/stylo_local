from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pipeline_common import GSECT, merge_analysis_inputs, present_structure_metrics, present_style_metrics


def _project(df: pd.DataFrame, cols: list[str], n_components: int = 5) -> np.ndarray:
    if not cols:
        raise RuntimeError("No columns available for PCA projection.")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, ~X.isna().all()]
    if X.shape[1] == 0:
        raise RuntimeError("All requested PCA columns are empty.")
    X_imp = SimpleImputer(strategy="mean").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
    return pca.fit_transform(X_scaled)


def main() -> None:
    df = merge_analysis_inputs()
    structure_cols = present_structure_metrics(df)
    style_cols = present_style_metrics(df)
    if not structure_cols:
        raise RuntimeError("No canonical structure metrics available for decoupling analysis.")
    if not style_cols:
        raise RuntimeError("No explicit style metrics available for decoupling analysis.")

    journals = df["journal_label"].astype(str).values
    style_df = df[style_cols].apply(pd.to_numeric, errors="coerce")
    style_df = style_df.loc[:, ~style_df.isna().all()]
    style_imp = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(style_df), columns=style_df.columns)

    style_res = style_imp.copy()
    for journal in pd.unique(journals):
        idx = journals == journal
        style_res.loc[idx] = style_imp.loc[idx] - style_imp.loc[idx].mean()

    structure_pc = _project(df, structure_cols)
    style_pc = _project(style_res, list(style_res.columns))

    corr = np.corrcoef(structure_pc.T, style_pc.T)[: structure_pc.shape[1], structure_pc.shape[1] :]
    corr_df = pd.DataFrame(
        corr,
        index=[f"struct_PC{i + 1}" for i in range(structure_pc.shape[1])],
        columns=[f"style_resid_PC{i + 1}" for i in range(style_pc.shape[1])],
    )

    out = GSECT / "structure_style_pc_correlations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out, index=True)

    print("Saved:", out)
    print("Structure columns:", len(structure_cols))
    print("Style columns:", len(style_res.columns))
    print("Correlation matrix (structure vs residual style PCs):")
    print(corr_df.round(3))


if __name__ == "__main__":
    main()
