from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import re
from datetime import datetime

HOME = Path.home()
STYLO_OUT = HOME / "stylo_local" / "stylo_out"
GROBID_TEI = STYLO_OUT / "grobid_tei"
GROBID_SECTIONS = STYLO_OUT / "grobid_sections"

CSV_RE = re.compile(r"\.csv$", re.IGNORECASE)
JSONL_RE = re.compile(r"\.jsonl$", re.IGNORECASE)
XML_RE = re.compile(r"\.xml$", re.IGNORECASE)
ZIP_RE = re.compile(r"\.zip$", re.IGNORECASE)
PARQUET_RE = re.compile(r"\.parquet$", re.IGNORECASE)


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024
    return f"{x:.1f}TB"


def count_lines_fast(path: Path) -> int | None:
    # Only for plain text files
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def file_meta(path: Path) -> dict:
    st = path.stat()
    meta = {
        "path": str(path),
        "relpath": str(path.relative_to(STYLO_OUT)) if path.is_relative_to(STYLO_OUT) else str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "bytes": st.st_size,
        "size_human": human_size(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }

    if CSV_RE.search(path.name) or JSONL_RE.search(path.name) or path.suffix.lower() in {".md", ".txt"}:
        n = count_lines_fast(path)
        meta["lines"] = n
        if n is not None and CSV_RE.search(path.name):
            meta["rows_est"] = max(n - 1, 0)
        else:
            meta["rows_est"] = None
    else:
        meta["lines"] = None
        meta["rows_est"] = None

    return meta


def classify(relpath: str) -> str:
    # coarse buckets for your pipeline outputs
    rp = relpath.lower()

    if rp.endswith(".tei.xml") or "/grobid_tei/" in rp:
        return "grobid_tei"
    if "/grobid_sections/" in rp:
        if "sections.jsonl" in rp:
            return "sections_text_store"
        if "per_article_section_metrics" in rp:
            return "section_metrics"
        if "section_name_frequencies" in rp:
            return "section_name_stats"
        if "master" in rp:
            return "master_table"
        if "journal_template_strength" in rp or "journal_section_template" in rp:
            return "journal_templates"
        if "eta2" in rp or "effect_sizes" in rp:
            return "journal_effect_sizes"
        if "author_signature" in rp or "clusters" in rp:
            return "residual_style_space"
        if "structure_style" in rp or "decoupling" in rp:
            return "structure_style_decoupling"
        if "metadata_enriched" in rp:
            return "metadata_enriched"
        return "grobid_sections_misc"

    if "per_article_metrics" in rp:
        return "article_metrics"
    if "bundles_top20_long" in rp:
        return "lexical_bundles"
    if rp.endswith(".zip"):
        return "archives"
    if rp.endswith(".parquet"):
        return "parquet"
    return "other"


def main():
    if not STYLO_OUT.exists():
        raise FileNotFoundError(f"Missing stylo_out dir: {STYLO_OUT}")

    all_files = []
    for p in STYLO_OUT.rglob("*"):
        if p.is_file():
            meta = file_meta(p)
            meta["category"] = classify(meta["relpath"])
            all_files.append(meta)

    # sort by category then name
    all_files.sort(key=lambda d: (d["category"], d["relpath"]))

    # Required outputs checklist (based on your current pipeline)
    required = [
        ("article_metrics", "per_article_metrics.csv"),
        ("article_metrics", "per_article_metrics.jsonl"),
        ("lexical_bundles", "bundles_top20_long.csv"),
        ("article_metrics", "per_article_metrics_analysis_ready.csv"),
        ("grobid_tei", "grobid_tei/ (folder with .tei.xml files)"),
        ("sections_text_store", "grobid_sections/sections.jsonl"),
        ("section_metrics", "grobid_sections/per_article_section_metrics_wide.csv"),
        ("section_name_stats", "grobid_sections/section_name_frequencies.csv"),
    ]

    # build quick lookup
    relnames = set([d["relpath"] for d in all_files])

    def has_pattern(pat: str) -> bool:
        if pat.endswith("/ (folder with .tei.xml files)"):
            return GROBID_TEI.exists() and any(GROBID_TEI.glob("*.tei.xml"))
        # exact match within relpath set
        return pat in relnames

    checklist = []
    for cat, pat in required:
        checklist.append({
            "required_item": pat,
            "category": cat,
            "present": has_pattern(pat),
        })

    # Write CSV inventory
    out_csv = STYLO_OUT / "OUTPUT_INVENTORY.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["category", "relpath", "size_human", "bytes", "lines", "rows_est", "mtime"]
        )
        w.writeheader()
        for d in all_files:
            w.writerow({
                "category": d["category"],
                "relpath": d["relpath"],
                "size_human": d["size_human"],
                "bytes": d["bytes"],
                "lines": d["lines"],
                "rows_est": d["rows_est"],
                "mtime": d["mtime"],
            })

    # Write Markdown report
    out_md = STYLO_OUT / "OUTPUT_INVENTORY.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# Stylo Output Inventory\n\n")
        f.write(f"- Root: `{STYLO_OUT}`\n")
        f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Total files: {len(all_files)}\n\n")

        f.write("## Required outputs checklist\n\n")
        for item in checklist:
            status = "OK" if item["present"] else "MISSING"
            f.write(f"- {status}: `{item['required_item']}` ({item['category']})\n")
        f.write("\n")

        # Summary counts by category
        f.write("## Summary by category\n\n")
        cat_counts = {}
        cat_bytes = {}
        for d in all_files:
            cat_counts[d["category"]] = cat_counts.get(d["category"], 0) + 1
            cat_bytes[d["category"]] = cat_bytes.get(d["category"], 0) + d["bytes"]

        for cat in sorted(cat_counts.keys()):
            f.write(f"- **{cat}**: {cat_counts[cat]} files, {human_size(cat_bytes[cat])}\n")
        f.write("\n")

        # Detailed listing per category
        f.write("## Detailed listing\n\n")
        current = None
        for d in all_files:
            if d["category"] != current:
                current = d["category"]
                f.write(f"### {current}\n\n")
            extra = ""
            if d["rows_est"] is not None:
                extra = f", rows≈{d['rows_est']}"
            elif d["lines"] is not None:
                extra = f", lines={d['lines']}"
            f.write(f"- `{d['relpath']}` ({d['size_human']}{extra})\n")

        f.write("\n## Notes\n\n")
        f.write("- `rows≈` for CSV is a fast estimate: (lines - 1). It is accurate unless the file has embedded newlines.\n")
        f.write("- TEI count is inferred by counting `*.tei.xml` in `grobid_tei/`.\n")

    print("Saved inventory reports:")
    print("-", out_csv)
    print("-", out_md)
    print("\nOpen them in Cursor to review.")


if __name__ == "__main__":
    main()