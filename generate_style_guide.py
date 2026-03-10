from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUT_DIR = Path.home() / "stylo_local" / "stylo_out"

STYLE_METRICS = [
    "avg_sentence_len",
    "median_sentence_len",
    "sd_sentence_len",
    "mtld",
    "passive_sent_ratio",
    "nominalisations_per_1k_words",
    "subordination_per_1k_words",
    "pos_adj_ratio",
    "pos_verb_ratio",
    "pos_noun_ratio",
    "pos_adv_ratio",
    "citations_per_1k",
    "integral_ratio",
    "sent_gt_40_pct",
    "sent_lt_12_pct",
]

METRIC_LABELS: dict[str, str] = {
    "avg_sentence_len": "Avg sentence length (words)",
    "median_sentence_len": "Median sentence length (words)",
    "sd_sentence_len": "Sentence length std dev",
    "mtld": "Lexical diversity (MTLD)",
    "passive_sent_ratio": "Passive sentence ratio",
    "nominalisations_per_1k_words": "Nominalisations per 1k words",
    "subordination_per_1k_words": "Subordination per 1k words",
    "pos_adj_ratio": "Adjective ratio",
    "pos_verb_ratio": "Verb ratio",
    "pos_noun_ratio": "Noun ratio",
    "pos_adv_ratio": "Adverb ratio",
    "citations_per_1k": "Citations per 1k words",
    "integral_ratio": "Integral citation ratio",
    "sent_gt_40_pct": "% sentences > 40 words",
    "sent_lt_12_pct": "% sentences < 12 words",
    "agentless_passive_ratio_of_passives": "Agentless passive ratio (of passives)",
}

# η² thresholds for journal influence classification
ETA_HIGH = 0.20
ETA_MID = 0.10

MIN_ARTICLES = 2


# ---------------------------------------------------------------------------
# CLI & Logging
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a per-journal writing style guide from pipeline outputs."
    )
    p.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory containing the pipeline CSV outputs (default: ~/stylo_local/stylo_out).",
    )
    p.add_argument(
        "--output_file",
        default="",
        help="Path for the output Markdown file. Defaults to <out_dir>/STYLE_GUIDE.md.",
    )
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four pipeline CSV files and return them as DataFrames."""

    per_article_path = out_dir / "per_article_metrics.csv"
    metadata_path = out_dir / "grobid_sections" / "metadata_enriched.csv"
    variance_path = out_dir / "journal_variance_analysis.csv"
    sections_path = out_dir / "canonical_sections_wide.csv"

    for p in (per_article_path, metadata_path, variance_path, sections_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")

    logging.info("Loading per_article_metrics from %s", per_article_path)
    metrics_df = pd.read_csv(per_article_path)

    logging.info("Loading metadata_enriched from %s", metadata_path)
    metadata_df = pd.read_csv(metadata_path)

    logging.info("Loading journal_variance_analysis from %s", variance_path)
    variance_df = pd.read_csv(variance_path)

    logging.info("Loading canonical_sections_wide from %s", sections_path)
    sections_df = pd.read_csv(sections_path)

    return metrics_df, metadata_df, variance_df, sections_df


# ---------------------------------------------------------------------------
# Data Merging
# ---------------------------------------------------------------------------


def _derive_file_stem(file_col: str) -> str:
    """Strip .pdf (case-insensitive) from the file column value."""
    s = str(file_col)
    if s.lower().endswith(".pdf"):
        return s[:-4]
    return s


def merge_data(
    metrics_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge metrics, metadata, and sections into one working DataFrame."""

    # Derive file_stem from the 'file' column in metrics
    metrics_df = metrics_df.copy()
    metrics_df["file_stem"] = metrics_df["file"].apply(_derive_file_stem)

    # Resolve journal: prefer journal_crossref, fall back to journal column
    metadata_df = metadata_df.copy()
    if "journal_crossref" in metadata_df.columns:
        # Use journal_crossref when journal is missing
        fallback = (
            metadata_df["journal"] if "journal" in metadata_df.columns else pd.Series(dtype=str)
        )
        metadata_df["journal_resolved"] = metadata_df["journal_crossref"].fillna(fallback)
    else:
        metadata_df["journal_resolved"] = (
            metadata_df["journal"] if "journal" in metadata_df.columns else pd.Series(dtype=str)
        )

    # Resolve year: prefer year column, fall back to crossref if available
    if "year" in metadata_df.columns:
        metadata_df["year_resolved"] = metadata_df["year"]
    else:
        metadata_df["year_resolved"] = pd.Series(dtype=str)

    # Select only needed metadata columns
    meta_cols = ["file_stem", "journal_resolved", "year_resolved"]
    meta_slim = metadata_df[[c for c in meta_cols if c in metadata_df.columns]].copy()

    # Merge metrics with metadata
    merged = metrics_df.merge(meta_slim, on="file_stem", how="left")

    # Merge with sections
    merged = merged.merge(sections_df, on="file_stem", how="left")

    logging.info("Merged DataFrame: %d articles, %d columns", len(merged), len(merged.columns))
    return merged


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _stats(series: pd.Series) -> dict[str, float]:
    """Return descriptive statistics for a numeric series, ignoring NaN."""
    clean = series.dropna()
    if len(clean) == 0:
        return {k: np.nan for k in ("mean", "median", "std", "p25", "p75", "min", "max")}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _fmt(value: float, decimals: int = 2) -> str:
    """Format a float, returning '–' for NaN."""
    if np.isnan(value):
        return "–"
    return f"{value:.{decimals}f}"


def _pct(value: float) -> str:
    """Format a 0–1 ratio as a percentage string."""
    if np.isnan(value):
        return "–"
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Journal Fingerprint helpers
# ---------------------------------------------------------------------------


def _classify_metric(eta_sq: float, p_value: float) -> tuple[str, str]:
    """Return (emoji, label) for a metric based on η² and p-value."""
    if eta_sq > ETA_HIGH and p_value < 0.05:
        return "🔴", "Journal-enforced"
    if eta_sq >= ETA_MID and p_value < 0.05:
        return "🟡", "Journal-influenced"
    return "🟢", "Author's voice"


def _build_fingerprint_map(variance_df: pd.DataFrame) -> dict[str, tuple[str, str, float, float]]:
    """Return {metric_name: (emoji, label, eta_sq, p_value)} for all metrics."""
    result: dict[str, tuple[str, str, float, float]] = {}
    for _, row in variance_df.iterrows():
        metric = str(row["metric_name"])
        eta = float(row["eta_sq"]) if not pd.isna(row["eta_sq"]) else 0.0
        pval = float(row["p_value"]) if not pd.isna(row["p_value"]) else 1.0
        emoji, label = _classify_metric(eta, pval)
        result[metric] = (emoji, label, eta, pval)
    return result


# ---------------------------------------------------------------------------
# Section analysis helpers
# ---------------------------------------------------------------------------


def _section_columns(df: pd.DataFrame) -> list[str]:
    """Return all section word-count or sentence-length columns present in df."""
    # Works with both old (BODY, OTHER) and new (ARGUMENT, FRAMEWORK) naming
    possible = [
        "INTRO_mtld", "INTRO_sentence_length",
        "FRAMEWORK_mtld", "FRAMEWORK_sentence_length",
        "ARGUMENT_mtld", "ARGUMENT_sentence_length",
        "BODY_mtld", "BODY_sentence_length",
        "DISCUSSION_mtld", "DISCUSSION_sentence_length",
        "CONCLUSION_mtld", "CONCLUSION_sentence_length",
        "OTHER_mtld", "OTHER_sentence_length",
    ]
    return [c for c in possible if c in df.columns]


def _section_presence(group_df: pd.DataFrame, section_cols: list[str]) -> dict[str, float]:
    """Return fraction of articles that have a non-NaN value for each section column."""
    result: dict[str, float] = {}
    for col in section_cols:
        if col in group_df.columns:
            n_present = group_df[col].notna().sum()
            result[col] = n_present / len(group_df)
    return result


# ---------------------------------------------------------------------------
# Writer's Takeaway generator
# ---------------------------------------------------------------------------


def _writers_takeaway(
    journal_stats: dict[str, dict[str, float]],
    corpus_stats: dict[str, dict[str, float]],
    fingerprint: dict[str, tuple[str, str, float, float]],
    n_articles: int,
    all_journal_citation_means: dict[str, float] | None = None,
) -> str:
    """Generate a plain-English paragraph summarising the journal's style profile."""
    lines: list[str] = []

    def _above(metric: str, threshold: float = 0.0) -> bool:
        j = journal_stats.get(metric, {}).get("mean", float("nan"))
        c = corpus_stats.get(metric, {}).get("mean", float("nan"))
        if np.isnan(j) or np.isnan(c):
            return False
        corpus_std = corpus_stats.get(metric, {}).get("std", 1.0)
        min_diff = corpus_std * 0.3
        return j > c + max(threshold, min_diff)

    def _below(metric: str, threshold: float = 0.0) -> bool:
        j = journal_stats.get(metric, {}).get("mean", float("nan"))
        c = corpus_stats.get(metric, {}).get("mean", float("nan"))
        if np.isnan(j) or np.isnan(c):
            return False
        corpus_std = corpus_stats.get(metric, {}).get("std", 1.0)
        min_diff = corpus_std * 0.3
        return j < c - max(threshold, min_diff)

    def _jval(metric: str) -> float:
        return journal_stats.get(metric, {}).get("mean", float("nan"))

    def _cval(metric: str) -> float:
        return corpus_stats.get(metric, {}).get("mean", float("nan"))

    # Sentence length
    if _above("avg_sentence_len"):
        lines.append(
            f"This journal favours **longer, more complex sentences** "
            f"(avg {_fmt(_jval('avg_sentence_len'))} words vs corpus avg {_fmt(_cval('avg_sentence_len'))})."
        )
    elif _below("avg_sentence_len"):
        lines.append(
            f"This journal favours **shorter, more direct sentences** "
            f"(avg {_fmt(_jval('avg_sentence_len'))} words vs corpus avg {_fmt(_cval('avg_sentence_len'))})."
        )

    # Lexical diversity
    if _above("mtld"):
        lines.append(
            f"**Vocabulary variety is valued** — aim for high lexical diversity "
            f"(MTLD {_fmt(_jval('mtld'))} vs corpus avg {_fmt(_cval('mtld'))})."
        )
    elif _below("mtld"):
        lines.append(
            f"Lexical diversity is **below the corpus average** "
            f"(MTLD {_fmt(_jval('mtld'))} vs {_fmt(_cval('mtld'))}); focused, technical vocabulary is the norm."
        )

    # Passive voice
    if _above("passive_sent_ratio"):
        lines.append(
            f"**Passive constructions are more common** here than average "
            f"({_pct(_jval('passive_sent_ratio'))} vs {_pct(_cval('passive_sent_ratio'))})."
        )
    elif _below("passive_sent_ratio"):
        lines.append(
            f"The journal leans **active voice** "
            f"({_pct(_jval('passive_sent_ratio'))} passive vs corpus avg {_pct(_cval('passive_sent_ratio'))})."
        )

    # Nominalisations
    if _above("nominalisations_per_1k_words"):
        lines.append(
            f"**Nominalisation is above average** ({_fmt(_jval('nominalisations_per_1k_words'))} per 1k words); "
            "the academic register is dense and noun-heavy."
        )

    # Citations
    fp_cite = fingerprint.get("citations_per_1k")
    if fp_cite and fp_cite[0] == "🔴":
        j_cite = _jval("citations_per_1k")
        c_cite = _cval("citations_per_1k")
        if not np.isnan(j_cite) and not np.isnan(c_cite) and all_journal_citation_means:
            sorted_vals = sorted(all_journal_citation_means.values())
            if len(sorted_vals) >= 2 and j_cite <= sorted_vals[0]:
                rank_note = " — the **lowest** in the corpus"
            elif len(sorted_vals) >= 2 and j_cite >= sorted_vals[-1]:
                rank_note = " — the **highest** in the corpus"
            else:
                rank_note = ""
            if j_cite < c_cite:
                lines.append(
                    f"This journal uses **very few citations** "
                    f"({_fmt(j_cite)}/1k vs corpus avg {_fmt(c_cite)}){rank_note} — "
                    f"match this journal's lean citation style."
                )
            else:
                lines.append(
                    f"This journal **cites heavily** "
                    f"({_fmt(j_cite)}/1k vs corpus avg {_fmt(c_cite)}){rank_note} — "
                    f"match this journal's citation density exactly."
                )
        else:
            lines.append(
                "**Citation density is strongly journal-driven** — match the journal's citation norms carefully."
            )
    elif _above("citations_per_1k"):
        lines.append(
            f"**Citations are more frequent** than average "
            f"({_fmt(_jval('citations_per_1k'))} vs {_fmt(_cval('citations_per_1k'))} per 1k words)."
        )
    elif _below("citations_per_1k"):
        lines.append(
            f"**Citations are lighter** than the corpus average "
            f"({_fmt(_jval('citations_per_1k'))} vs {_fmt(_cval('citations_per_1k'))} per 1k words)."
        )

    # Integral citations
    if _above("integral_ratio"):
        lines.append(
            "**Integral citations** (citing authors as grammatical subjects) are preferred here "
            f"({_pct(_jval('integral_ratio'))} of all citations)."
        )

    # Short sentences
    if _above("sent_lt_12_pct"):
        lines.append(
            f"A higher proportion of **short sentences (< 12 words)** "
            f"({_pct(_jval('sent_lt_12_pct'))} vs {_pct(_cval('sent_lt_12_pct'))}) suggests a punchy, varied rhythm."
        )

    # Long sentences
    if _above("sent_gt_40_pct"):
        lines.append(
            f"A notable proportion of **very long sentences (> 40 words)** "
            f"({_pct(_jval('sent_gt_40_pct'))} vs {_pct(_cval('sent_gt_40_pct'))}) — complex, multi-clause constructions are welcome."
        )

    # Fallback
    if not lines:
        lines.append(
            f"This journal's style is broadly **in line with the corpus average** across all measured features "
            f"(based on {n_articles} articles)."
        )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Key Findings & Cross-Journal Contrasts generators
# ---------------------------------------------------------------------------


def _md_key_findings(
    variance_df: pd.DataFrame,
    all_journal_stats: dict[str, dict[str, dict[str, float]]],
    corpus_stats: dict[str, dict[str, float]],
    fingerprint: dict[str, tuple[str, str, float, float]],
) -> str:
    """Generate an auto-computed Key Findings narrative section."""
    lines: list[str] = ["\n## 🎯 Key Findings\n"]
    findings: list[str] = []
    covered_metrics: set[str] = set()

    sorted_var = variance_df.dropna(subset=["eta_sq"]).sort_values("eta_sq", ascending=False)

    # Finding 1: Metric with the highest η²
    if len(sorted_var) > 0:
        top_row = sorted_var.iloc[0]
        top_metric = str(top_row["metric_name"])
        top_eta = float(top_row["eta_sq"])
        top_label = METRIC_LABELS.get(top_metric, top_metric)
        covered_metrics.add(top_metric)

        journal_means: dict[str, float] = {}
        for journal, stats in all_journal_stats.items():
            v = stats.get(top_metric, {}).get("mean", float("nan"))
            if not np.isnan(v):
                journal_means[journal] = v

        if len(journal_means) >= 2:
            low_j = min(journal_means, key=journal_means.__getitem__)
            high_j = max(journal_means, key=journal_means.__getitem__)
            low_v = journal_means[low_j]
            high_v = journal_means[high_j]

            if top_metric == "citations_per_1k":
                if low_v > 0:
                    ratio = high_v / low_v
                    finding = (
                        f"**{top_label} is almost entirely journal-determined** (η² = {top_eta:.2f}). "
                        f"{high_j} expects {high_v:.2f} citations per 1,000 words — "
                        f"{ratio:.0f}× more than {low_j} ({low_v:.2f}). "
                        f"Match your target journal's citation density exactly."
                    )
                else:
                    finding = (
                        f"**{top_label} is almost entirely journal-determined** (η² = {top_eta:.2f}). "
                        f"{high_j} expects {high_v:.2f} citations per 1,000 words vs {low_j} ({low_v:.2f})."
                    )
            elif top_metric in ("sent_lt_12_pct", "sent_gt_40_pct", "passive_sent_ratio", "integral_ratio"):
                finding = (
                    f"**{top_label} is almost entirely journal-determined** (η² = {top_eta:.2f}), "
                    f"ranging from {low_v * 100:.0f}% in {low_j} to {high_v * 100:.0f}% in {high_j}."
                )
            else:
                finding = (
                    f"**{top_label} is strongly journal-determined** (η² = {top_eta:.2f}), "
                    f"ranging from {low_v:.2f} in {low_j} to {high_v:.2f} in {high_j}."
                )
            findings.append(finding)

    # Finding 2: Largest cross-journal spread among remaining metrics
    best_spread = -1.0
    best_metric = ""
    best_low_j = best_high_j = ""
    best_low_v = best_high_v = 0.0

    for metric in STYLE_METRICS:
        if metric in covered_metrics:
            continue
        c_std = corpus_stats.get(metric, {}).get("std", float("nan"))
        if np.isnan(c_std) or c_std == 0:
            continue
        journal_means = {}
        for journal, stats in all_journal_stats.items():
            v = stats.get(metric, {}).get("mean", float("nan"))
            if not np.isnan(v):
                journal_means[journal] = v
        if len(journal_means) < 2:
            continue
        lj = min(journal_means, key=journal_means.__getitem__)
        hj = max(journal_means, key=journal_means.__getitem__)
        spread = (journal_means[hj] - journal_means[lj]) / c_std
        if spread > best_spread:
            best_spread = spread
            best_metric = metric
            best_low_j, best_high_j = lj, hj
            best_low_v, best_high_v = journal_means[lj], journal_means[hj]

    if best_metric:
        covered_metrics.add(best_metric)
        label = METRIC_LABELS.get(best_metric, best_metric)
        fp = fingerprint.get(best_metric, ("🟢", "Author's voice", 0.0, 1.0))
        if best_metric == "mtld":
            finding = (
                f"**Vocabulary diversity separates journals**: {best_high_j} demands the richest vocabulary "
                f"(MTLD {best_high_v:.2f}) while {best_low_j} uses more focused, repetitive terminology "
                f"(MTLD {best_low_v:.2f})."
            )
        elif best_metric in ("sent_lt_12_pct", "sent_gt_40_pct", "passive_sent_ratio", "integral_ratio"):
            finding = (
                f"**{label} shows notable cross-journal variation** (η² = {fp[2]:.2f}): "
                f"from {best_low_v * 100:.0f}% in {best_low_j} to {best_high_v * 100:.0f}% in {best_high_j}."
            )
        else:
            finding = (
                f"**{label} shows notable cross-journal variation**: from {best_low_v:.2f} in "
                f"{best_low_j} to {best_high_v:.2f} in {best_high_j}."
            )
        findings.append(finding)

    # Finding 3: Next journal-influenced metric not yet covered
    for _, row in sorted_var.iterrows():
        metric = str(row["metric_name"])
        if metric in covered_metrics:
            continue
        fp = fingerprint.get(metric, ("🟢", "Author's voice", 0.0, 1.0))
        if fp[0] not in ("🔴", "🟡"):
            continue
        eta = float(row["eta_sq"])
        label = METRIC_LABELS.get(metric, metric)
        journal_means = {}
        for journal, stats in all_journal_stats.items():
            v = stats.get(metric, {}).get("mean", float("nan"))
            if not np.isnan(v):
                journal_means[journal] = v
        if len(journal_means) < 2:
            continue
        lj = min(journal_means, key=journal_means.__getitem__)
        hj = max(journal_means, key=journal_means.__getitem__)
        lv = journal_means[lj]
        hv = journal_means[hj]
        if metric in ("sent_lt_12_pct", "sent_gt_40_pct", "passive_sent_ratio", "integral_ratio"):
            finding = (
                f"**{label} is journal-influenced** (η² = {eta:.2f}): "
                f"from {lv * 100:.0f}% in {lj} to {hv * 100:.0f}% in {hj}."
            )
        else:
            finding = (
                f"**{label} is journal-influenced** (η² = {eta:.2f}): "
                f"from {lv:.2f} in {lj} to {hv:.2f} in {hj}."
            )
        findings.append(finding)
        break

    for i, finding in enumerate(findings[:3], 1):
        lines.append(f"{i}. {finding}\n")

    if not findings:
        lines.append("_No key findings could be computed from the available data._\n")

    return "\n".join(lines)


def _md_cross_journal_contrasts(
    all_journal_stats: dict[str, dict[str, dict[str, float]]],
    corpus_stats: dict[str, dict[str, float]],
) -> str:
    """Generate a cross-journal comparison table showing the biggest metric differences."""
    lines: list[str] = ["\n## 🔍 Key Cross-Journal Contrasts\n"]
    lines.append("The biggest differences between journals in this corpus:\n")

    contrasts: list[tuple[float, str, str, float, str, float, str]] = []

    for metric in STYLE_METRICS:
        c_std = corpus_stats.get(metric, {}).get("std", float("nan"))
        if np.isnan(c_std) or c_std == 0:
            continue
        journal_means: dict[str, float] = {}
        for journal, stats in all_journal_stats.items():
            v = stats.get(metric, {}).get("mean", float("nan"))
            if not np.isnan(v):
                journal_means[journal] = v
        if len(journal_means) < 2:
            continue
        low_j = min(journal_means, key=journal_means.__getitem__)
        high_j = max(journal_means, key=journal_means.__getitem__)
        low_v = journal_means[low_j]
        high_v = journal_means[high_j]
        spread = (high_v - low_v) / c_std

        if metric == "citations_per_1k" and low_v > 0:
            spread_str = f"{high_v / low_v:.0f}×"
        elif metric in ("passive_sent_ratio", "integral_ratio", "sent_gt_40_pct", "sent_lt_12_pct"):
            spread_str = f"+{(high_v - low_v) * 100:.0f}pp"
        else:
            spread_str = f"+{high_v - low_v:.2f}"

        contrasts.append((spread, metric, low_j, low_v, high_j, high_v, spread_str))

    contrasts.sort(key=lambda x: x[0], reverse=True)

    rows: list[str] = [
        "| Metric | Lowest Journal | Highest Journal | Spread |",
        "|--------|---------------|-----------------|--------|",
    ]
    for _, metric, low_j, low_v, high_j, high_v, spread_str in contrasts[:5]:
        label = METRIC_LABELS.get(metric, metric)
        if metric in ("passive_sent_ratio", "integral_ratio", "sent_gt_40_pct", "sent_lt_12_pct"):
            low_fmt = f"{low_v * 100:.0f}%"
            high_fmt = f"{high_v * 100:.0f}%"
        else:
            low_fmt = f"{low_v:.2f}"
            high_fmt = f"{high_v:.2f}"
        rows.append(f"| {label} | {low_j} ({low_fmt}) | {high_j} ({high_fmt}) | {spread_str} |")

    lines.append("\n".join(rows))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def _md_stats_table(
    metrics: list[str],
    journal_stats: dict[str, dict[str, float]],
    corpus_stats: dict[str, dict[str, float]],
    fingerprint: dict[str, tuple[str, str, float, float]],
) -> str:
    rows: list[str] = [
        "| Metric | Mean | Median | Std | P25–P75 | Corpus Mean | Influence |",
        "|--------|------|--------|-----|---------|-------------|-----------|",
    ]
    for m in metrics:
        if m not in journal_stats:
            continue
        js = journal_stats[m]
        cs = corpus_stats.get(m, {})
        fp = fingerprint.get(m, ("🟢", "Author's voice", 0.0, 1.0))
        label = f"{fp[0]} {fp[1]}"
        rows.append(
            f"| {METRIC_LABELS.get(m, m)} "
            f"| {_fmt(js['mean'])} "
            f"| {_fmt(js['median'])} "
            f"| {_fmt(js['std'])} "
            f"| {_fmt(js['p25'])}–{_fmt(js['p75'])} "
            f"| {_fmt(cs.get('mean', float('nan')))} "
            f"| {label} |"
        )
    return "\n".join(rows)


def _md_section_table(presence: dict[str, float], group_df: pd.DataFrame) -> str:
    if not presence:
        return "_No section data available._"

    rows: list[str] = [
        "| Section | % Articles | Avg MTLD | Avg Sentence Length |",
        "|---------|-----------|----------|---------------------|",
    ]
    # Group section columns by section name using the first underscore as delimiter
    # e.g. "INTRO_mtld" → "INTRO", "CONCLUSION_sentence_length" → "CONCLUSION"
    sections_seen: dict[str, list[str]] = {}
    for col in presence:
        section = col.split("_", 1)[0]
        sections_seen.setdefault(section, []).append(col)

    for section, cols in sorted(sections_seen.items()):
        pct_vals = [presence[c] for c in cols if c in presence]
        pct = max(pct_vals) if pct_vals else float("nan")

        mtld_col = f"{section}_mtld"
        sent_col = f"{section}_sentence_length"
        avg_mtld: float = float("nan")
        avg_sent: float = float("nan")
        if mtld_col in group_df.columns:
            avg_mtld = float(group_df[mtld_col].mean())
        if sent_col in group_df.columns:
            avg_sent = float(group_df[sent_col].mean())

        rows.append(
            f"| {section} "
            f"| {pct * 100:.0f}% "
            f"| {_fmt(avg_mtld)} "
            f"| {_fmt(avg_sent)} |"
        )
    return "\n".join(rows)


def _md_journal_profile(
    journal: str,
    group_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    fingerprint: dict[str, tuple[str, str, float, float]],
    section_cols: list[str],
    all_journal_citation_means: dict[str, float] | None = None,
) -> str:
    n = len(group_df)
    available_metrics = [m for m in STYLE_METRICS if m in group_df.columns]

    # Compute per-journal stats
    journal_stats = {m: _stats(group_df[m]) for m in available_metrics}
    corpus_stats = {m: _stats(corpus_df[m]) for m in available_metrics}

    # Article shape
    total_words_stats = _stats(group_df["total_words"]) if "total_words" in group_df.columns else {}
    total_sents_stats = _stats(group_df["total_sentences"]) if "total_sentences" in group_df.columns else {}

    # Section presence
    presence = _section_presence(group_df, section_cols)

    # Takeaway
    takeaway = _writers_takeaway(
        journal_stats, corpus_stats, fingerprint, n, all_journal_citation_means
    )

    lines: list[str] = []
    lines.append(f"\n---\n\n## 📖 {journal}\n")
    lines.append(f"**Articles in corpus:** {n}\n")

    # Article shape
    lines.append("\n### 📐 Article Shape\n")
    if total_words_stats:
        lines.append(
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Avg word count | {_fmt(total_words_stats.get('mean', float('nan')), 0)} |\n"
            f"| Word count range | {_fmt(total_words_stats.get('min', float('nan')), 0)}–{_fmt(total_words_stats.get('max', float('nan')), 0)} |\n"
            f"| Avg sentence count | {_fmt(total_sents_stats.get('mean', float('nan')), 0)} |\n"
        )

    # Prose style profile
    lines.append("\n### ✍️ Prose Style Profile\n")
    lines.append(_md_stats_table(available_metrics, journal_stats, corpus_stats, fingerprint))

    # Section template
    lines.append("\n\n### 🏗️ Structure Template\n")
    lines.append(_md_section_table(presence, group_df))

    # Writer's takeaway
    lines.append("\n\n### 💡 Writer's Takeaway\n")
    lines.append(f"> {takeaway}\n")

    return "\n".join(lines)


def _md_eta_table(variance_df: pd.DataFrame) -> str:
    fingerprint = _build_fingerprint_map(variance_df)
    sorted_df = variance_df.sort_values("eta_sq", ascending=False)

    rows: list[str] = [
        "| Metric | η² | p-value | F-stat | Influence Level | Interpretation |",
        "|--------|----|---------|--------|-----------------|----------------|",
    ]
    interp_map = {
        "🔴": "You MUST match this journal's norm",
        "🟡": "Moderate journal norm — worth aligning with",
        "🟢": "Express yourself here — this is your author voice",
    }
    for _, row in sorted_df.iterrows():
        metric = str(row["metric_name"])
        eta = float(row["eta_sq"]) if not pd.isna(row["eta_sq"]) else 0.0
        pval = float(row["p_value"]) if not pd.isna(row["p_value"]) else 1.0
        fstat = float(row["f_stat"]) if not pd.isna(row["f_stat"]) else 0.0
        fp = fingerprint.get(metric, ("🟢", "Author's voice", 0.0, 1.0))
        label = f"{fp[0]} {fp[1]}"
        interp = interp_map.get(fp[0], "")
        rows.append(
            f"| {METRIC_LABELS.get(metric, metric)} "
            f"| {eta:.3f} "
            f"| {pval:.4g} "
            f"| {fstat:.2f} "
            f"| {label} "
            f"| {interp} |"
        )
    return "\n".join(rows)


def _md_corpus_overview(
    merged: pd.DataFrame,
    corpus_stats: dict[str, dict[str, float]],
) -> str:
    lines: list[str] = []
    lines.append("\n## 🌍 Corpus-Wide Patterns\n")
    lines.append(
        "These values represent the **overall mean** across all articles and can serve as a baseline "
        "when no journal-specific target is available.\n"
    )

    rows: list[str] = [
        "| Metric | Corpus Mean | Corpus Median | Std |",
        "|--------|-------------|---------------|-----|",
    ]
    for m in STYLE_METRICS:
        if m not in corpus_stats:
            continue
        cs = corpus_stats[m]
        rows.append(
            f"| {METRIC_LABELS.get(m, m)} "
            f"| {_fmt(cs['mean'])} "
            f"| {_fmt(cs['median'])} "
            f"| {_fmt(cs['std'])} |"
        )
    lines.append("\n".join(rows))
    return "\n".join(lines)


def _md_checklist() -> str:
    return """
## ✅ Practical Writing Checklist

Use this checklist before submitting to your target journal:

**Style & Readability**
- [ ] Check your average sentence length against the journal's target range
- [ ] Review sentences > 40 words — can any be split without losing nuance?
- [ ] Check sentences < 12 words — are they intentional for emphasis?
- [ ] Measure your text's MTLD score (lexical diversity) against the journal norm

**Grammar & Register**
- [ ] Count your passive constructions — do they match the journal's ratio?
- [ ] Scan for excessive nominalisations (e.g. "examination of" instead of "examining")
- [ ] Review subordination density — too many embedded clauses can obscure your argument

**Citations**
- [ ] Count your citations per 1,000 words and compare to journal norm
- [ ] Check your integral vs non-integral citation ratio (do you cite *Smith (2020) argues* or *(Smith, 2020)*?)
- [ ] Ensure citation style matches the 🔴 Journal-enforced benchmark above

**Structure**
- [ ] Verify your article has the canonical sections for the target journal
- [ ] Check section order matches the journal's typical template
- [ ] Ensure your introduction and conclusion are proportionate in length

**Final Check**
- [ ] Re-read the journal's Writer's Takeaway section above
- [ ] Compare your article's key metrics to the journal profile table
- [ ] Note any 🔴 Journal-enforced features and prioritise matching those
"""


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


def generate_report(
    merged: pd.DataFrame,
    variance_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Generate the full STYLE_GUIDE.md Markdown report."""

    fingerprint = _build_fingerprint_map(variance_df)
    section_cols = _section_columns(merged)

    # Resolve journal column
    journal_col = "journal_resolved" if "journal_resolved" in merged.columns else "journal"
    if journal_col not in merged.columns:
        raise RuntimeError(
            f"No journal column found in merged data. "
            f"Expected 'journal_resolved' or 'journal'. Available: {list(merged.columns)}"
        )

    # Drop articles with no journal assignment
    valid = merged[merged[journal_col].notna()].copy()
    if len(valid) == 0:
        raise RuntimeError("No articles have a journal assignment — cannot generate profiles.")

    logging.info("Articles with journal assignment: %d / %d", len(valid), len(merged))

    journal_counts = valid[journal_col].value_counts()
    eligible_journals = journal_counts[journal_counts >= MIN_ARTICLES].index.tolist()
    singleton_journals = journal_counts[journal_counts < MIN_ARTICLES].index.tolist()

    logging.info(
        "Journals with >= %d articles: %d; singletons skipped: %d",
        MIN_ARTICLES,
        len(eligible_journals),
        len(singleton_journals),
    )

    # Corpus-wide stats (across all articles with a journal)
    available_metrics = [m for m in STYLE_METRICS if m in valid.columns]
    corpus_stats = {m: _stats(valid[m]) for m in available_metrics}

    # Build per-journal stats for cross-journal comparison functions
    all_journal_stats: dict[str, dict[str, dict[str, float]]] = {}
    for journal in eligible_journals:
        group_df = valid[valid[journal_col] == journal]
        all_journal_stats[journal] = {m: _stats(group_df[m]) for m in available_metrics}

    # Convenience dict: journal → mean citations_per_1k (for writer's takeaway)
    all_journal_citation_means: dict[str, float] = {}
    for journal, jstats in all_journal_stats.items():
        mean_val = jstats.get("citations_per_1k", {}).get("mean", float("nan"))
        if not np.isnan(mean_val):
            all_journal_citation_means[journal] = mean_val

    # Year range
    year_series = valid.get("year_resolved", pd.Series(dtype=str))
    years_numeric = pd.to_numeric(year_series, errors="coerce").dropna()
    if len(years_numeric) > 0:
        year_range = f"{int(years_numeric.min())}–{int(years_numeric.max())}"
    else:
        year_range = "unknown"

    # --------------- Build the Markdown output ---------------
    parts: list[str] = []

    # Header
    parts.append("# 📚 Journal Writing Style Guide\n")
    parts.append(
        "_Auto-generated by `generate_style_guide.py` from the Stylo pipeline outputs._\n"
    )

    # Executive Summary
    parts.append("\n## 📊 Executive Summary\n")
    parts.append(
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Total articles analysed | {len(valid)} |\n"
        f"| Journals with profiles | {len(eligible_journals)} |\n"
        f"| Singletons (skipped) | {len(singleton_journals)} |\n"
        f"| Year range | {year_range} |\n"
        f"| Metrics analysed | {len(available_metrics)} |\n"
    )
    parts.append(
        "\n> **How to read this guide:** Each journal profile shows mean/median/std and P25–P75 "
        "for key stylometric features, compared against the full corpus average. "
        "The **Influence** column uses η² from ANOVA to classify each metric as "
        "🔴 Journal-enforced, 🟡 Journal-influenced, or 🟢 Author's voice.\n"
    )

    # Key Findings
    parts.append(
        _md_key_findings(variance_df, all_journal_stats, corpus_stats, fingerprint)
    )

    # Corpus overview
    parts.append(_md_corpus_overview(valid, corpus_stats))

    # Cross-journal contrasts
    parts.append(_md_cross_journal_contrasts(all_journal_stats, corpus_stats))

    # Per-journal profiles (sorted by article count, most first)
    parts.append("\n---\n\n# 🗂️ Per-Journal Profiles\n")
    for journal in eligible_journals:
        group_df = valid[valid[journal_col] == journal].copy()
        profile_md = _md_journal_profile(
            journal=journal,
            group_df=group_df,
            corpus_df=valid,
            variance_df=variance_df,
            fingerprint=fingerprint,
            section_cols=section_cols,
            all_journal_citation_means=all_journal_citation_means,
        )
        parts.append(profile_md)

    # Singletons note
    if singleton_journals:
        parts.append("\n---\n\n## ℹ️ Skipped Journals (fewer than 2 articles)\n")
        parts.append(
            "The following journals had only one article in the corpus and were excluded from profiling:\n"
        )
        for j in sorted(singleton_journals):
            parts.append(f"- {j}\n")

    # η² insight table
    parts.append("\n---\n\n## 🔬 The η² Insight Table\n")
    parts.append(
        "This table shows which stylometric features are **driven by the journal** "
        "(high η²) vs which belong to the **author's individual voice** (low η²).\n\n"
        "_Sorted by effect size (η²) — highest journal influence first._\n\n"
    )
    parts.append(_md_eta_table(variance_df))

    # Practical checklist
    parts.append(_md_checklist())

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    logging.info("Style guide written to: %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    output_file = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else out_dir / "STYLE_GUIDE.md"
    )

    logging.info("Reading pipeline outputs from: %s", out_dir)
    metrics_df, metadata_df, variance_df, sections_df = load_data(out_dir)

    merged = merge_data(metrics_df, metadata_df, sections_df)

    generate_report(merged, variance_df, output_file)
    logging.info("Done. Open %s to read the style guide.", output_file)


if __name__ == "__main__":
    main()
