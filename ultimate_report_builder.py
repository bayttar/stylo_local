from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


HOME = Path.home()
BASE_OUT = HOME / "stylo_local" / "stylo_out"
SECT_OUT = BASE_OUT / "grobid_sections"

MASTER = SECT_OUT / "MASTER_v2.csv"

# Inputs we expect (some optional)
RESID_CLUST = SECT_OUT / "author_signature_residual_clusters.csv"
JOUR_TPL_STRENGTH = SECT_OUT / "journal_template_strength.csv"
ETA2 = SECT_OUT / "journal_effect_sizes_eta2.csv"
STRUCT_STYLE_CORR = SECT_OUT / "structure_style_pc_correlations.csv"  # global matrix (optional)

# Outputs
OUT_RAW_CSV = BASE_OUT / "ULTIMATE_ACADEMIC_STYLO_REPORT_RAW_v1.csv"
OUT_ENR_CSV = BASE_OUT / "ULTIMATE_ACADEMIC_STYLO_REPORT_ENRICHED_v1.csv"
OUT_RAW_PARQ = BASE_OUT / "ULTIMATE_ACADEMIC_STYLO_REPORT_RAW_v1.parquet"
OUT_ENR_PARQ = BASE_OUT / "ULTIMATE_ACADEMIC_STYLO_REPORT_ENRICHED_v1.parquet"

# For detecting section-share columns
SEC_SHARE_SUFFIX = "_word_share"


def _safe_read_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Missing required file: {path}")
    return None


def _normalise_file_stem_from_filecol(df: pd.DataFrame, file_col: str = "file") -> pd.DataFrame:
    if file_col in df.columns and "file_stem" not in df.columns:
        df = df.copy()
        df["file_stem"] = df[file_col].astype(str).str.replace(r"\.pdf$", "", regex=True)
    return df


def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    x = df.select_dtypes(include=[np.number]).copy()
    if x.shape[1] == 0:
        return x
    return x.astype(float)


def _drop_all_nan_cols(X: pd.DataFrame) -> pd.DataFrame:
    if X.shape[1] == 0:
        return X
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        print("Dropping all-NaN numeric columns:", all_nan)
        X = X.drop(columns=all_nan)
    return X


def _impute_mean(X: pd.DataFrame) -> pd.DataFrame:
    if X.shape[1] == 0:
        return X
    imp = SimpleImputer(strategy="mean")
    arr = imp.fit_transform(X)
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def _compute_structure_pcs(df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    share_cols = [c for c in df.columns if c.endswith(SEC_SHARE_SUFFIX)]
    if not share_cols:
        return pd.DataFrame(index=df.index)

    S = df[share_cols].astype(float).fillna(0.0)
    S_scaled = StandardScaler().fit_transform(S)

    k = min(n_components, S_scaled.shape[1])
    pca = PCA(n_components=k, random_state=42)
    pcs = pca.fit_transform(S_scaled)

    out = pd.DataFrame(pcs, columns=[f"struct_PC{i+1}" for i in range(k)], index=df.index)
    # extra structure summaries
    out["structure_section_entropy_bits_article"] = _row_entropy_bits(S.values)
    out["structure_dominant_section_share_article"] = np.max(S.values, axis=1)
    out["structure_nonzero_section_count_article"] = (S.values > 0).sum(axis=1)
    return out


def _row_entropy_bits(mat: np.ndarray) -> np.ndarray:
    # Shannon entropy in bits, row-wise
    mat = np.array(mat, dtype=float)
    ent = np.zeros(mat.shape[0], dtype=float)
    for i in range(mat.shape[0]):
        p = mat[i]
        p = p[p > 0]
        if p.size == 0:
            ent[i] = 0.0
        else:
            ent[i] = float(-(p * np.log2(p)).sum())
    return ent


def _compute_residual_style_pcs(df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    # Exclude structure shares and obvious identifiers; keep numeric stylometry
    share_cols = [c for c in df.columns if c.endswith(SEC_SHARE_SUFFIX)]
    drop = set(share_cols)
    drop |= {
        "file", "file_stem", "title", "doi", "authors",
        "journal_label", "journal_crossref", "journal",
        "publisher", "publisher_crossref",
    }

    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = _select_numeric(X)
    X = _drop_all_nan_cols(X)
    X = _impute_mean(X)

    if X.shape[1] == 0:
        return pd.DataFrame(index=df.index)

    journals = df.get("journal_label", pd.Series(["UNKNOWN"] * len(df), index=df.index)).fillna("UNKNOWN").astype(str)

    # journal de-mean
    X_res = X.copy()
    for j in journals.unique():
        idx = (journals == j)
        mu = X.loc[idx].mean()
        X_res.loc[idx] = X.loc[idx] - mu

    X_scaled = StandardScaler().fit_transform(X_res)
    k = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=k, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    out = pd.DataFrame(pcs, columns=[f"style_resid_PC{i+1}" for i in range(k)], index=df.index)

    # distance from journal centroid in residual space (first k PCs)
    dist = np.zeros(len(df), dtype=float)
    for j in journals.unique():
        idx = (journals == j).to_numpy()
        cent = pcs[idx].mean(axis=0)
        dist[idx] = np.linalg.norm(pcs[idx] - cent, axis=1)
    out["style_resid_dist_from_journal_centroid"] = dist
    return out


def _load_eta2_weights(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    e = pd.read_csv(path)
    if "feature" not in e.columns or "eta_squared_journal" not in e.columns:
        return None
    e = e[["feature", "eta_squared_journal"]].dropna()
    e["eta_squared_journal"] = e["eta_squared_journal"].astype(float)
    return e.sort_values("eta_squared_journal", ascending=False)


def _constraint_scores_from_eta2(df: pd.DataFrame, eta2: pd.DataFrame | None) -> pd.DataFrame:
    """
    Enriched-only: creates corpus-level constraint indices derived from eta².
    Not a behaviour change to existing metrics; adds synthetic indices.

    - journal_constraint_score_weighted: sum(|z(feature)| * eta2(feature)) over intersecting features
    - journal_constraint_score_top50: same but restricted to top50 eta² features
    - journal_constraint_top_features: JSON list of top10 eta² feature names (global, not per-article)
    """
    out = pd.DataFrame(index=df.index)
    if eta2 is None or eta2.empty:
        return out

    # Candidate numeric columns in df (exclude structure shares + IDs)
    share_cols = [c for c in df.columns if c.endswith(SEC_SHARE_SUFFIX)]
    drop = set(share_cols)
    drop |= {"file", "file_stem", "title", "doi", "authors", "journal_label"}

    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = _select_numeric(X)
    X = _drop_all_nan_cols(X)
    X = _impute_mean(X)

    if X.shape[1] == 0:
        return out

    # Align weights to available columns
    w = eta2.set_index("feature")["eta_squared_journal"]
    common = [c for c in X.columns if c in w.index]
    if not common:
        return out

    Z = StandardScaler().fit_transform(X[common].values)
    absZ = np.abs(Z)
    weights = w.loc[common].values.reshape(1, -1)

    score_all = (absZ * weights).sum(axis=1)
    out["journal_constraint_score_weighted"] = score_all

    top50 = eta2.head(50)["feature"].tolist()
    top50_common = [c for c in common if c in set(top50)]
    if top50_common:
        Z50 = StandardScaler().fit_transform(X[top50_common].values)
        absZ50 = np.abs(Z50)
        w50 = w.loc[top50_common].values.reshape(1, -1)
        out["journal_constraint_score_top50"] = (absZ50 * w50).sum(axis=1)
    else:
        out["journal_constraint_score_top50"] = np.nan

    out["journal_constraint_top_features"] = json.dumps(eta2.head(10)["feature"].tolist(), ensure_ascii=False)
    out["journal_constraint_eta2_mean"] = float(eta2["eta_squared_journal"].mean())
    out["journal_constraint_eta2_max"] = float(eta2["eta_squared_journal"].max())
    return out


def _merge_optional(base: pd.DataFrame, other: pd.DataFrame | None, on: str, how: str = "left") -> pd.DataFrame:
    if other is None or other.empty:
        return base
    if on not in base.columns or on not in other.columns:
        return base
    return base.merge(other, on=on, how=how)


def main():
    # --- Load base (already includes article metrics + section shares + crossref fields) ---
    df = _safe_read_csv(MASTER, required=True)
    df = _normalise_file_stem_from_filecol(df, "file")

    # --- Attach residual clusters (optional but expected) ---
    cl = _safe_read_csv(RESID_CLUST, required=False)
    if cl is not None:
        cl = _normalise_file_stem_from_filecol(cl, "file")
        df = _merge_optional(df, cl[["file", "cluster"]].rename(columns={"cluster": "residual_cluster_id"}), on="file", how="left")

    # --- Attach journal template strength (journal-level broadcast) ---
    jts = _safe_read_csv(JOUR_TPL_STRENGTH, required=False)
    if jts is not None and "journal_label" in df.columns and "journal_label" in jts.columns:
        # Keep only stable, journal-level columns
        keep = ["journal_label"]
        for c in ["section_entropy_bits", "dominant_section_share"]:
            if c in jts.columns:
                keep.append(c)
        jts = jts[keep].copy()
        # rename to avoid confusion with article-level entropy
        ren = {}
        if "section_entropy_bits" in jts.columns:
            ren["section_entropy_bits"] = "journal_section_entropy_bits"
        if "dominant_section_share" in jts.columns:
            ren["dominant_section_share"] = "journal_dominant_section_share"
        jts = jts.rename(columns=ren)
        df = df.merge(jts, on="journal_label", how="left")

    # --- Compute structure PCs and article-level structure summaries ---
    struct = _compute_structure_pcs(df, n_components=5)
    df_raw = df.copy()
    df_raw = pd.concat([df_raw, struct], axis=1)

    # --- Compute residual style PCs and distance summaries ---
    style = _compute_residual_style_pcs(df, n_components=5)
    df_raw = pd.concat([df_raw, style], axis=1)

    # --- Save RAW (maximal, mostly “everything we have”) ---
    df_raw.to_csv(OUT_RAW_CSV, index=False)
    try:
        df_raw.to_parquet(OUT_RAW_PARQ, index=False)
    except Exception as e:
        print("Parquet write failed (RAW):", e)

    print("Saved RAW:")
    print("-", OUT_RAW_CSV)
    print("-", OUT_RAW_PARQ)

    # --- ENRICHED: add synthetic meta-indices (constraint scores, compact summaries) ---
    eta2 = _load_eta2_weights(ETA2)
    enriched = _constraint_scores_from_eta2(df_raw, eta2)

    df_enr = df_raw.copy()
    df_enr = pd.concat([df_enr, enriched], axis=1)

    # Add a global decoupling summary if correlation matrix exists
    if STRUCT_STYLE_CORR.exists():
        try:
            corr = pd.read_csv(STRUCT_STYLE_CORR, index_col=0)
            # summarise absolute correlation magnitude
            df_enr["structure_style_decoupling_max_abs_corr_global"] = float(np.nanmax(np.abs(corr.values)))
            df_enr["structure_style_decoupling_mean_abs_corr_global"] = float(np.nanmean(np.abs(corr.values)))
        except Exception:
            df_enr["structure_style_decoupling_max_abs_corr_global"] = np.nan
            df_enr["structure_style_decoupling_mean_abs_corr_global"] = np.nan

    df_enr.to_csv(OUT_ENR_CSV, index=False)
    try:
        df_enr.to_parquet(OUT_ENR_PARQ, index=False)
    except Exception as e:
        print("Parquet write failed (ENRICHED):", e)

    print("Saved ENRICHED:")
    print("-", OUT_ENR_CSV)
    print("-", OUT_ENR_PARQ)

    # Minimal run summary
    print("\nRUN SUMMARY")
    print("Rows:", len(df_enr))
    print("Cols (RAW):", df_raw.shape[1])
    print("Cols (ENR):", df_enr.shape[1])
    if "journal_label" in df_enr.columns:
        print("Unique journals:", df_enr["journal_label"].nunique(dropna=True))


if __name__ == "__main__":
    main()