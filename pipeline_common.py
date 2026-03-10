from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path.home() / "stylo_local" / "stylo_out"
GSECT = ROOT / "grobid_sections"

STYLE_METRICS_ALLOWLIST = [
    "mtld",
    "avg_sentence_len",
    "median_sentence_len",
    "sd_sentence_len",
    "sent_gt_40_pct",
    "sent_lt_12_pct",
    "subordination_per_1k_words",
    "nominalisations_per_1k_words",
    "passive_sent_ratio",
    "agentless_passive_ratio_of_passives",
    "citations_per_1k",
    "integral_ratio",
    "pos_noun_ratio",
    "pos_verb_ratio",
    "pos_adj_ratio",
    "pos_adv_ratio",
]

CANONICAL_STRUCTURE_ALLOWLIST = [
    "INTRO_mtld",
    "INTRO_sentence_length",
    "FRAMEWORK_mtld",
    "FRAMEWORK_sentence_length",
    "ARGUMENT_mtld",
    "ARGUMENT_sentence_length",
    "DISCUSSION_mtld",
    "DISCUSSION_sentence_length",
    "CONCLUSION_mtld",
    "CONCLUSION_sentence_length",
]

MISSING_MARKERS = {"", "missing", "na", "n/a", "none", "null", "nan", "unknown", "0"}


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def is_missing(value: object) -> bool:
    return clean_text(value).lower() in MISSING_MARKERS


def metrics_with_file_stem(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()
    if "file_stem" not in out.columns:
        if "file" not in out.columns:
            raise ValueError("Metrics CSV must contain 'file' or 'file_stem'.")
        out["file_stem"] = out["file"].astype(str).str.replace(r"\.pdf$", "", regex=True)
    return out


def metadata_with_file_stem(metadata_df: pd.DataFrame) -> pd.DataFrame:
    out = metadata_df.copy()
    if "file_stem" not in out.columns:
        if "tei_file" not in out.columns:
            raise ValueError("Metadata CSV must contain 'file_stem' or 'tei_file'.")
        out["file_stem"] = out["tei_file"].astype(str).str.replace(r"\.tei\.xml$", "", regex=True)
    return out


def resolve_journal_series(metadata_df: pd.DataFrame) -> pd.Series:
    if "journal" in metadata_df.columns:
        resolved = metadata_df["journal"]
    elif "journal_crossref" in metadata_df.columns:
        resolved = metadata_df["journal_crossref"]
    else:
        raise ValueError("Metadata CSV must contain a unified 'journal' column.")
    return resolved.astype(str).map(clean_text)


def resolve_publisher_series(metadata_df: pd.DataFrame) -> pd.Series:
    if "publisher" in metadata_df.columns:
        resolved = metadata_df["publisher"]
    elif "publisher_crossref" in metadata_df.columns:
        resolved = metadata_df["publisher_crossref"]
    else:
        resolved = pd.Series([""] * len(metadata_df), index=metadata_df.index)
    return resolved.astype(str).map(clean_text)


def load_metrics(metrics_path: Path | None = None) -> pd.DataFrame:
    path = metrics_path or (ROOT / "per_article_metrics.csv")
    return metrics_with_file_stem(pd.read_csv(path))


def load_metadata(metadata_path: Path | None = None) -> pd.DataFrame:
    path = metadata_path or (GSECT / "metadata_enriched.csv")
    meta = metadata_with_file_stem(pd.read_csv(path))
    meta["journal_label"] = resolve_journal_series(meta)
    meta["publisher_label"] = resolve_publisher_series(meta)
    return meta


def load_sections(sections_path: Path | None = None) -> pd.DataFrame:
    path = sections_path or (ROOT / "canonical_sections_wide.csv")
    df = pd.read_csv(path)
    if "file_stem" not in df.columns:
        raise ValueError("Canonical sections CSV must contain 'file_stem'.")
    return df


def merge_metrics_and_metadata(
    metrics_path: Path | None = None,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    metrics = load_metrics(metrics_path)
    meta = load_metadata(metadata_path)
    meta_small = meta[["file_stem", "journal_label", "publisher_label", "title", "doi"]].drop_duplicates(
        subset=["file_stem"]
    )
    merged = metrics.merge(meta_small, on="file_stem", how="left")
    merged = merged.loc[~merged["journal_label"].map(is_missing)].copy()
    if merged.empty:
        raise RuntimeError("No rows remained after merging metrics with resolved journal labels.")
    return merged


def merge_analysis_inputs(
    metrics_path: Path | None = None,
    metadata_path: Path | None = None,
    sections_path: Path | None = None,
) -> pd.DataFrame:
    merged = merge_metrics_and_metadata(metrics_path, metadata_path)
    sections = load_sections(sections_path)
    out = merged.merge(sections, on="file_stem", how="inner")
    if out.empty:
        raise RuntimeError("Merged analysis table is empty after joining canonical sections.")
    return out


def present_style_metrics(df: pd.DataFrame) -> list[str]:
    return [col for col in STYLE_METRICS_ALLOWLIST if col in df.columns]


def present_structure_metrics(df: pd.DataFrame) -> list[str]:
    return [col for col in CANONICAL_STRUCTURE_ALLOWLIST if col in df.columns]
