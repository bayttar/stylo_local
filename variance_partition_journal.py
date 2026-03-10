from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from pipeline_common import GSECT, merge_metrics_and_metadata, present_style_metrics


MIN_GROUP_SIZE = 2


def eta_squared_by_group(x: np.ndarray, g: np.ndarray) -> tuple[float, float, float] | None:
    x = np.asarray(x, dtype=float)
    g = np.asarray(g)

    mask = np.isfinite(x)
    x = x[mask]
    g = g[mask]
    if x.size == 0:
        return None

    grouped: list[np.ndarray] = []
    for label in pd.unique(g):
        vals = x[g == label]
        if vals.size < MIN_GROUP_SIZE:
            return None
        if np.isclose(np.var(vals, ddof=1), 0.0):
            return None
        grouped.append(vals)

    if len(grouped) < 2:
        return None

    grand_mean = float(np.mean(x))
    ss_between = 0.0
    ss_within = 0.0
    for vals in grouped:
        mu = float(np.mean(vals))
        ss_between += float(vals.size) * ((mu - grand_mean) ** 2)
        ss_within += float(np.sum((vals - mu) ** 2))

    ss_total = ss_between + ss_within
    if np.isclose(ss_total, 0.0):
        return None

    f_stat, p_value = f_oneway(*grouped)
    eta_sq = float(np.clip(ss_between / ss_total, 0.0, 1.0))
    return eta_sq, float(f_stat), float(p_value)


def main() -> None:
    df = merge_metrics_and_metadata()
    metric_cols = present_style_metrics(df)
    if not metric_cols:
        raise RuntimeError("No explicit style metrics available for eta-squared analysis.")

    journals = df["journal_label"].astype(str).values
    rows: list[dict[str, float | str | bool]] = []
    skipped = 0
    for col in metric_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        stats = eta_squared_by_group(values, journals)
        if stats is None:
            skipped += 1
            continue
        eta_sq, f_stat, p_value = stats
        rows.append(
            {
                "metric_name": col,
                "eta_sq": eta_sq,
                "f_stat": f_stat,
                "p_value": p_value,
                "is_significant": p_value < 0.05,
            }
        )

    if not rows:
        raise RuntimeError("No analyzable metrics survived the explicit style-metric allowlist.")

    out = GSECT / "journal_effect_sizes_eta2.csv"
    res = pd.DataFrame(rows).sort_values("eta_sq", ascending=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    print("Saved:", out)
    print("Computed metrics:", len(res))
    print("Skipped metrics:", skipped)
    print()
    print("Top journal-driven features (highest eta^2):")
    print(res.head(25))


if __name__ == "__main__":
    main()
