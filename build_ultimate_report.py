from __future__ import annotations

from pathlib import Path
import json
import math
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from pipeline_common import clean_text, is_missing

# Optional plotting (graceful fallback)
PLOTTING_OK = True
try:
    import matplotlib.pyplot as plt
except Exception:
    PLOTTING_OK = False

ROOT = Path.home() / "stylo_local" / "stylo_out"
GSECT = ROOT / "grobid_sections"

# Core inputs (expected from your pipeline)
PATHS = {
    # Article-level stylometry
    "per_article_metrics": ROOT / "per_article_metrics.csv",
    "per_article_metrics_jsonl": ROOT / "per_article_metrics.jsonl",
    "bundles_long": ROOT / "bundles_top20_long.csv",
    "per_article_metrics_analysis_ready": ROOT / "per_article_metrics_analysis_ready.csv",

    # GROBID-derived
    "sections_jsonl": GSECT / "sections.jsonl",
    "per_article_section_wide": ROOT / "canonical_sections_wide.csv",
    "section_name_frequencies": GSECT / "section_name_frequencies.csv",
    "section_raw_name_counts": GSECT / "section_raw_name_counts.csv",
    "template_complexity": GSECT / "per_article_template_complexity.csv",
    "journal_template_strength": GSECT / "journal_template_strength.csv",
    "journal_section_template_canonical": GSECT / "journal_section_template_canonical.csv",

    # Metadata enrichment
    "metadata_from_tei": GSECT / "metadata_from_tei.csv",
    "metadata_enriched": GSECT / "metadata_enriched.csv",

    # Residual / variance / decoupling outputs
    "author_signature_clusters": GSECT / "author_signature_residual_clusters.csv",
    "eta2": ROOT / "journal_variance_analysis.csv",
    "structure_style_pc_corr": GSECT / "structure_style_pc_correlations.csv",
}

# Output files
OUT_MD = ROOT / "ULTIMATE_REPORT.md"
OUT_HTML = ROOT / "ULTIMATE_REPORT.html"
OUT_FIG_DIR = ROOT / "ultimate_report_figures"


# -------------------------
# Utilities
# -------------------------
def must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {label}\nExpected: {path}")

def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def human_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{100*x:.1f}%"

def fmt_float(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"

def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_(empty)_\n"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)

    # Prefer pandas to_markdown if tabulate exists, else fallback to a simple pipe table.
    try:
        return d.to_markdown(index=False) + "\n"
    except Exception:
        # Minimal markdown table fallback (no tabulate dependency)
        cols = list(d.columns)
        # Convert to strings, avoid newlines
        rows = []
        for _, r in d.iterrows():
            row = []
            for c in cols:
                v = r[c]
                if pd.isna(v):
                    s = "NA"
                else:
                    s = str(v)
                s = s.replace("\n", " ").replace("|", "\\|")
                row.append(s)
            rows.append(row)

        header = "| " + " | ".join(cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        body = "".join(["| " + " | ".join(row) + " |\n" for row in rows])
        return header + sep + body + "\n"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_hist(series: pd.Series, title: str, fname: str) -> Optional[str]:
    if not PLOTTING_OK:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    ensure_dir(OUT_FIG_DIR)
    out = OUT_FIG_DIR / fname
    plt.figure()
    plt.hist(s.values, bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return str(out.relative_to(ROOT))

def save_bar_counts(counts: pd.Series, title: str, fname: str, topn: int = 20) -> Optional[str]:
    if not PLOTTING_OK:
        return None
    c = counts.dropna()
    if c.empty:
        return None
    c = c.sort_values(ascending=False).head(topn)
    ensure_dir(OUT_FIG_DIR)
    out = OUT_FIG_DIR / fname
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(c)), c.values)
    plt.xticks(range(len(c)), c.index, rotation=70, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return str(out.relative_to(ROOT))

def file_inventory_snapshot(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            rows.append({
                "relpath": str(p.relative_to(root)),
                "bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            })
    df = pd.DataFrame(rows).sort_values(["relpath"])
    return df

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def resolve_metadata_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "journal" in out.columns:
        out["journal_resolved"] = out["journal"].astype(str).map(clean_text)
    elif "journal_crossref" in out.columns:
        out["journal_resolved"] = out["journal_crossref"].astype(str).map(clean_text)
    else:
        out["journal_resolved"] = ""

    if "publisher" in out.columns:
        out["publisher_resolved"] = out["publisher"].astype(str).map(clean_text)
    elif "publisher_crossref" in out.columns:
        out["publisher_resolved"] = out["publisher_crossref"].astype(str).map(clean_text)
    else:
        out["publisher_resolved"] = ""
    return out


# -------------------------
# Report sections
# -------------------------
def section_A_article_level(per_article: pd.DataFrame, bundles_long: pd.DataFrame) -> str:
    # Key metrics list (must match your intent)
    keys = [
        "avg_sentence_len",
        "sd_sentence_len",
        "sent_gt_40_pct",
        "sent_lt_12_pct",
        "subordination_per_1k_words",
        "nominalisations_per_1k_words",
        "passive_sent_ratio",
        "agentless_passive_ratio_of_passives",
        "mtld",
        "pos_noun_ratio",
        "pos_verb_ratio",
        "pos_adj_ratio",
        "pos_adv_ratio",
        "citations_per_1k",
        "integral_ratio",
        "quote_count",
        "quote_len_mean",
        "block_quote_starts",
        "emdash_per_1k_words",
    ]
    present = [c for c in keys if c in per_article.columns]
    df = coerce_numeric(per_article, present)

    # Summary stats
    desc = df[present].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    desc = desc.reset_index().rename(columns={"index": "metric"})
    desc = desc[["metric", "count", "mean", "std", "min", "10%", "25%", "50%", "75%", "90%", "max"]]

    # Sentence bins (if stored as JSON string in analysis_ready, here may be object-like; keep simple)
    # Identify bin count columns if your CSV expanded them; otherwise skip.
    bin_cols = [c for c in per_article.columns if c.startswith("sentence_bin_counts")]
    # likely not expanded; we do not force-parsing here.

    # Lexical bundles: global top 30 across corpus
    top_bundles = pd.DataFrame()
    if {"ngram", "count"}.issubset(bundles_long.columns):
        b = bundles_long.copy()
        b["count"] = pd.to_numeric(b["count"], errors="coerce").fillna(0).astype(int)
        top_bundles = (
            b.groupby("ngram", as_index=False)["count"].sum()
            .sort_values("count", ascending=False)
            .head(30)
        )

    figs = []
    figs.append(save_hist(df.get("avg_sentence_len"), "Average sentence length (words)", "A_avg_sentence_len.png"))
    figs.append(save_hist(df.get("subordination_per_1k_words"), "Subordination per 1k words", "A_subordination_per_1k.png"))
    figs.append(save_hist(df.get("mtld"), "MTLD (lexical diversity)", "A_mtld.png"))
    figs.append(save_hist(df.get("citations_per_1k"), "Citations per 1k words", "A_citations_per_1k.png"))
    figs = [f for f in figs if f]

    out = []
    out.append("## A) Article-level deep stylometry (structure-aware)\n")
    out.append("This section summarises the corpus-wide distribution of core stylometric and academic-structure metrics extracted per article.\n")

    if figs:
        out.append("### A.1 Quick distribution plots\n")
        for f in figs:
            out.append(f"- `{f}`\n")
        out.append("\n")

    out.append("### A.2 Corpus-level descriptive statistics (selected metrics)\n")
    out.append(md_table(desc, max_rows=200))

    if not top_bundles.empty:
        out.append("### A.3 Lexical bundles (global top 30 across corpus)\n")
        out.append(md_table(top_bundles, max_rows=30))
    else:
        out.append("### A.3 Lexical bundles\n_(bundles_top20_long.csv missing or unexpected schema)_\n")

    return "".join(out)


def section_B_section_level(section_wide: pd.DataFrame,
                           section_name_freq: pd.DataFrame,
                           template_complexity: pd.DataFrame,
                           journal_canonical: pd.DataFrame,
                           journal_strength: pd.DataFrame) -> str:
    out = []
    out.append("## B) Section-level structure metrics (GROBID-derived)\n")
    out.append("This section reports section ecology: section shares, canonical mapping, entropy/template complexity, and journal-level templates.\n")

    # Section name frequencies
    if not section_name_freq.empty:
        cols = [c for c in ["section_name", "count"] if c in section_name_freq.columns]
        df = section_name_freq.copy()
        # attempt typical naming
        if "section_name" not in df.columns:
            # guess the first string column
            str_cols = [c for c in df.columns if df[c].dtype == "object"]
            if str_cols:
                df = df.rename(columns={str_cols[0]: "section_name"})
        if "count" not in df.columns:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df = df.rename(columns={num_cols[0]: "count"})
        if "count" in df.columns:
            df["count"] = pd.to_numeric(df["count"], errors="coerce")
        top = df.sort_values("count", ascending=False).head(30)[["section_name", "count"]]
        fig = save_bar_counts(top.set_index("section_name")["count"], "Top section names (raw/canonical mix)", "B_section_names_top.png", topn=20)
        if fig:
            out.append("### B.1 Section name distribution\n")
            out.append(f"- `{fig}`\n\n")
        out.append("Top 30 section labels:\n")
        out.append(md_table(top, max_rows=30))
    else:
        out.append("### B.1 Section name distribution\n_(section_name_frequencies.csv missing or empty)_\n")

    # Section wide metrics basic schema sanity
    out.append("### B.2 Section-level metrics table (wide) sanity\n")
    out.append(f"- Rows: {len(section_wide)}\n")
    out.append(f"- Columns: {section_wide.shape[1]}\n\n")

    # Template complexity / entropy if present
    out.append("### B.3 Per-article template complexity / entropy\n")
    if not template_complexity.empty:
        show_cols = [c for c in template_complexity.columns if c.lower() in {
            "file", "section_entropy", "entropy", "template_entropy", "template_complexity",
            "canonical_section_count", "section_count", "unique_section_count"
        }]
        if not show_cols:
            show_cols = template_complexity.columns[: min(12, template_complexity.shape[1])].tolist()
        out.append(md_table(template_complexity[show_cols].head(30), max_rows=30))
    else:
        out.append("_(per_article_template_complexity.csv missing or empty)_\n")

    # Journal-level canonical template
    out.append("### B.4 Canonical section mapping and journal template\n")
    if not journal_canonical.empty:
        out.append(md_table(journal_canonical, max_rows=200))
    else:
        out.append("_(journal_section_template_canonical.csv missing or empty)_\n")

    if not journal_strength.empty:
        out.append("### B.5 Journal template strength\n")
        out.append(md_table(journal_strength, max_rows=200))
    else:
        out.append("_(journal_template_strength.csv missing or empty)_\n")

    return "".join(out)


def section_C_metadata(metadata_enriched: pd.DataFrame, metadata_from_tei: pd.DataFrame) -> str:
    metadata_enriched = resolve_metadata_fields(metadata_enriched)
    metadata_from_tei = resolve_metadata_fields(metadata_from_tei) if not metadata_from_tei.empty else metadata_from_tei
    out = []
    out.append("## C) Journal and bibliographic metadata (TEI + DOI enrichment)\n")
    out.append("This section audits available metadata and reports coverage (non-null ratios) and top labels.\n")

    def coverage(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        rows = []
        for c in cols:
            if c in df.columns:
                present = ~df[c].astype(str).map(is_missing)
                rows.append({"field": c, "non_null_ratio": present.mean(), "non_null_count": int(present.sum())})
        return pd.DataFrame(rows).sort_values("non_null_ratio", ascending=False)

    # Pick typical metadata fields if present
    candidate_cols = [
        "file", "title", "doi", "journal_resolved", "publisher_resolved", "year",
        "authors", "author", "author_names", "issn", "eissn", "url"
    ]
    cov_en = coverage(metadata_enriched, candidate_cols)
    cov_tei = coverage(metadata_from_tei, candidate_cols)

    out.append("### C.1 Coverage (enriched metadata)\n")
    out.append(md_table(cov_en, max_rows=200))

    out.append("### C.2 Coverage (raw TEI metadata)\n")
    out.append(md_table(cov_tei, max_rows=200))

    # Top journals/publishers if present
    if "journal_resolved" in metadata_enriched.columns:
        top_j = metadata_enriched["journal_resolved"].replace("", "MISSING").value_counts().head(20).reset_index()
        top_j.columns = ["journal_resolved", "count"]
        out.append("### C.3 Top journal labels (enriched)\n")
        out.append(md_table(top_j, max_rows=20))

    if "publisher_resolved" in metadata_enriched.columns:
        top_p = metadata_enriched["publisher_resolved"].replace("", "MISSING").value_counts().head(20).reset_index()
        top_p.columns = ["publisher_resolved", "count"]
        out.append("### C.4 Top publishers (enriched)\n")
        out.append(md_table(top_p, max_rows=20))

    if "doi" in metadata_enriched.columns:
        nn = (~metadata_enriched["doi"].astype(str).map(is_missing)).mean()
        out.append(f"### C.5 DOI coverage\n- DOI non-null ratio: {human_pct(nn)}\n\n")

    return "".join(out)


def section_D_residual_space(author_clusters: pd.DataFrame) -> str:
    out = []
    out.append("## D) Residual stylometric space (journal mean removed, PCA, clustering)\n")
    out.append("This section summarises the residual space outputs already computed by your pipeline.\n")

    if author_clusters.empty:
        out.append("_(author_signature_residual_clusters.csv missing or empty)_\n")
        return "".join(out)

    # Detect cluster column
    cluster_col = None
    for c in author_clusters.columns:
        if c.lower() in {"cluster", "kmeans_cluster", "label"}:
            cluster_col = c
            break

    out.append("### D.1 Cluster distribution\n")
    if cluster_col:
        counts = author_clusters[cluster_col].value_counts().sort_index().reset_index()
        counts.columns = ["cluster", "count"]
        out.append(md_table(counts, max_rows=200))
    else:
        out.append("_(No obvious cluster column found. Columns are: " + ", ".join(author_clusters.columns[:20]) + ")_\n")

    out.append("### D.2 Sample rows\n")
    out.append(md_table(author_clusters.head(15), max_rows=15))

    return "".join(out)


def section_E_eta2(eta2: pd.DataFrame) -> str:
    out = []
    out.append("## E) Variance partition (η²): journal-driven vs individual metrics\n")
    out.append("This section ranks metrics by journal effect size (η²) and highlights the most journal-determined dimensions.\n")

    if eta2.empty:
        out.append("_(journal_variance_analysis.csv missing or empty)_\n")
        return "".join(out)

    # Try to infer columns
    cols = [c.lower() for c in eta2.columns]
    metric_col = None
    eta_col = None

    for c in eta2.columns:
        if c.lower() in {"metric", "feature", "variable"}:
            metric_col = c
    for c in eta2.columns:
        if c.lower() in {"eta2", "eta_squared", "etasq", "effect_size"}:
            eta_col = c

    if metric_col is None:
        # best guess: first object column
        obj_cols = [c for c in eta2.columns if eta2[c].dtype == "object"]
        metric_col = obj_cols[0] if obj_cols else eta2.columns[0]
    if eta_col is None:
        # best guess: first numeric column
        num_cols = [c for c in eta2.columns if pd.api.types.is_numeric_dtype(eta2[c])]
        eta_col = num_cols[0] if num_cols else eta2.columns[1]

    df = eta2.copy()
    df[eta_col] = pd.to_numeric(df[eta_col], errors="coerce")
    top = df.sort_values(eta_col, ascending=False).head(50)[[metric_col, eta_col]]
    top.columns = ["metric", "eta2"]

    out.append("### E.1 Top 50 journal-driven metrics (highest η²)\n")
    out.append(md_table(top, max_rows=50))

    # coarse buckets by prefix
    def bucket(m: str) -> str:
        m = str(m)
        if m.startswith("section_") or "section" in m.lower():
            return "section-structure"
        if "citation" in m.lower() or "quote" in m.lower():
            return "citation-ecology"
        if "pos_" in m.lower() or "fw_" in m.lower():
            return "pos/function-words"
        if "mtld" in m.lower() or "band_" in m.lower() or "nominal" in m.lower():
            return "lexical"
        if "passive" in m.lower() or "subordination" in m.lower():
            return "syntax"
        if "emdash" in m.lower() or "semicolon" in m.lower() or "colon" in m.lower():
            return "punctuation"
        return "other"

    top["bucket"] = top["metric"].apply(bucket)
    bucket_summary = top.groupby("bucket", as_index=False)["eta2"].mean().sort_values("eta2", ascending=False)
    out.append("### E.2 Mean η² by coarse metric family (top-50 only)\n")
    out.append(md_table(bucket_summary, max_rows=50))

    return "".join(out)


def section_F_decoupling(struct_style_corr: pd.DataFrame) -> str:
    out = []
    out.append("## F) Structure vs style decoupling (section PCs vs residual style PCs)\n")
    out.append("This section reports the correlation table produced by your decoupling step.\n")

    if struct_style_corr.empty:
        out.append("_(structure_style_pc_correlations.csv missing or empty)_\n")
        return "".join(out)

    out.append(md_table(struct_style_corr, max_rows=200))
    return "".join(out)


def section_G_journal_profiles(
    per_article: pd.DataFrame,
    metadata_enriched: pd.DataFrame,
    canonical_sections: pd.DataFrame,
) -> str:
    out = []
    out.append("## G) Per-journal profile summaries\n")
    out.append("This section keeps the most useful journal-facing view from the retired style-guide branch.\n")

    meta = resolve_metadata_fields(metadata_enriched)
    if "file_stem" not in meta.columns and "tei_file" in meta.columns:
        meta["file_stem"] = meta["tei_file"].astype(str).str.replace(r"\.tei\.xml$", "", regex=True)
    art = per_article.copy()
    if "file_stem" not in art.columns and "file" in art.columns:
        art["file_stem"] = art["file"].astype(str).str.replace(r"\.pdf$", "", regex=True)

    merged = art.merge(meta[["file_stem", "journal_resolved"]], on="file_stem", how="left")
    merged = merged.merge(canonical_sections, on="file_stem", how="left")
    merged = merged.loc[~merged["journal_resolved"].astype(str).map(is_missing)].copy()
    if merged.empty:
        out.append("_(No journals could be resolved for profile generation.)_\n")
        return "".join(out)

    rows = []
    for journal, group in merged.groupby("journal_resolved"):
        if len(group) < 2:
            continue
        rows.append(
            {
                "journal": journal,
                "articles": len(group),
                "avg_words": pd.to_numeric(group.get("total_words"), errors="coerce").mean(),
                "avg_mtld": pd.to_numeric(group.get("mtld"), errors="coerce").mean(),
                "avg_sentence_len": pd.to_numeric(group.get("avg_sentence_len"), errors="coerce").mean(),
                "avg_citations_per_1k": pd.to_numeric(group.get("citations_per_1k"), errors="coerce").mean(),
                "intro_presence": group.get("INTRO_mtld", pd.Series(dtype=float)).notna().mean(),
                "framework_presence": group.get("FRAMEWORK_mtld", pd.Series(dtype=float)).notna().mean(),
                "conclusion_presence": group.get("CONCLUSION_mtld", pd.Series(dtype=float)).notna().mean(),
            }
        )

    if not rows:
        out.append("_(No journals had at least 2 articles for profile generation.)_\n")
        return "".join(out)

    df = pd.DataFrame(rows).sort_values(["articles", "journal"], ascending=[False, True])
    out.append(md_table(df, max_rows=200))
    return "".join(out)


def appendix_inventory_and_quality(per_article: pd.DataFrame,
                                  section_wide: pd.DataFrame,
                                  metadata_enriched: pd.DataFrame) -> str:
    out = []
    out.append("## Appendix: Inventory and data quality audit\n")

    inv = file_inventory_snapshot(ROOT)
    out.append("### Inventory snapshot (files under stylo_out)\n")
    out.append(f"- Total files: {len(inv)}\n\n")
    out.append(md_table(inv.head(60), max_rows=60))

    # Quality checks
    out.append("### Data quality checks\n")

    out.append(f"- per_article_metrics rows: {len(per_article)}\n")
    out.append(f"- canonical_sections_wide rows: {len(section_wide)}\n")
    out.append(f"- metadata_enriched rows: {len(metadata_enriched)}\n\n")

    # Missingness highlights (top 25 columns by missing ratio)
    def missing_report(df: pd.DataFrame, label: str, topn: int = 25) -> str:
        if df.empty:
            return f"#### {label}\n_(empty)_\n"
        miss = df.isna().mean().sort_values(ascending=False).head(topn).reset_index()
        miss.columns = ["column", "missing_ratio"]
        return f"#### {label}\n" + md_table(miss, max_rows=topn)

    out.append(missing_report(metadata_enriched, "Metadata (enriched): missingness (top 25 columns)"))
    out.append(missing_report(section_wide, "Section-wide metrics: missingness (top 25 columns)"))
    out.append(missing_report(per_article, "Article metrics: missingness (top 25 columns)"))

    return "".join(out)


def build_html_from_md(md_text: str) -> str:
    # Minimal self-contained HTML. No external deps.
    # Markdown rendering: keep it simple with <pre> if no markdown lib.
    # If you want proper markdown->HTML, install "markdown" and we upgrade automatically.
    try:
        import markdown  # type: ignore
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Ultimate Academic Stylometry Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 40px; }}
code, pre {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
pre {{ padding: 12px; overflow-x: auto; }}
table {{ border-collapse: collapse; margin: 14px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }}
th {{ background: #f0f0f0; }}
h1, h2, h3 {{ margin-top: 26px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    except Exception:
        # fallback
        esc = (
            md_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Ultimate Academic Stylometry Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 40px; }}
pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; overflow-x: auto; }}
</style>
</head>
<body>
<pre>{esc}</pre>
</body>
</html>"""


def main():
    # Hard requirements (based on your inventory)
    must_exist(PATHS["per_article_metrics"], "per_article_metrics.csv")
    must_exist(PATHS["bundles_long"], "bundles_top20_long.csv")
    must_exist(PATHS["per_article_section_wide"], "canonical_sections_wide.csv")
    must_exist(PATHS["section_name_frequencies"], "section_name_frequencies.csv")
    must_exist(PATHS["metadata_enriched"], "metadata_enriched.csv")
    must_exist(PATHS["author_signature_clusters"], "author_signature_residual_clusters.csv")
    must_exist(PATHS["eta2"], "journal_variance_analysis.csv")
    must_exist(PATHS["structure_style_pc_corr"], "structure_style_pc_correlations.csv")

    # Load
    per_article = safe_read_csv(PATHS["per_article_metrics"])
    bundles_long = safe_read_csv(PATHS["bundles_long"])
    section_wide = safe_read_csv(PATHS["per_article_section_wide"])
    section_name_freq = safe_read_csv(PATHS["section_name_frequencies"])

    template_complexity = safe_read_csv(PATHS["template_complexity"]) if PATHS["template_complexity"].exists() else pd.DataFrame()
    journal_canonical = safe_read_csv(PATHS["journal_section_template_canonical"]) if PATHS["journal_section_template_canonical"].exists() else pd.DataFrame()
    journal_strength = safe_read_csv(PATHS["journal_template_strength"]) if PATHS["journal_template_strength"].exists() else pd.DataFrame()

    metadata_from_tei = safe_read_csv(PATHS["metadata_from_tei"]) if PATHS["metadata_from_tei"].exists() else pd.DataFrame()
    metadata_enriched = safe_read_csv(PATHS["metadata_enriched"])

    author_clusters = safe_read_csv(PATHS["author_signature_clusters"])
    eta2 = safe_read_csv(PATHS["eta2"])
    struct_style_corr = safe_read_csv(PATHS["structure_style_pc_corr"])

    # Build MD
    md = []
    md.append("# Ultimate Academic Stylometry Report (compiled from stylo_out)\n\n")
    md.append(f"- Root: `{ROOT}`\n")
    md.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    md.append(f"- Corpus size (articles): {len(per_article)}\n\n")

    md.append("## Executive overview\n")
    md.append(
        "This report consolidates outputs from: (A) article-level deep stylometry, "
        "(B) section-level structure metrics, (C) metadata enrichment, "
        "(D) residual stylometric space, (E) variance partition via η², "
        "(F) structure–style decoupling.\n\n"
    )

    md.append(section_A_article_level(per_article, bundles_long))
    md.append(section_B_section_level(section_wide, section_name_freq, template_complexity, journal_canonical, journal_strength))
    md.append(section_C_metadata(metadata_enriched, metadata_from_tei))
    md.append(section_D_residual_space(author_clusters))
    md.append(section_E_eta2(eta2))
    md.append(section_F_decoupling(struct_style_corr))
    md.append(section_G_journal_profiles(per_article, metadata_enriched, section_wide))
    md.append(appendix_inventory_and_quality(per_article, section_wide, metadata_enriched))

    md_text = "".join(md)

    OUT_MD.write_text(md_text, encoding="utf-8")

    # HTML (self-contained)
    html = build_html_from_md(md_text)
    OUT_HTML.write_text(html, encoding="utf-8")

    print("Saved:")
    print("-", OUT_MD)
    print("-", OUT_HTML)
    if PLOTTING_OK:
        print("-", OUT_FIG_DIR, "(if any plots were generated)")
    else:
        print("- plotting disabled (matplotlib not available)")

if __name__ == "__main__":
    main()
