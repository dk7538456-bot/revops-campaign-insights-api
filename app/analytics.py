"""Campaign data preparation and explainable analytics."""

from __future__ import annotations

from typing import Any

import pandas as pd
try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # Keep the deterministic IQR path usable without the optional ML package.
    IsolationForest = None

REQUIRED_COLUMNS = ["campaign", "spend", "impressions", "clicks", "conversions"]
METRIC_COLUMNS = ["spend", "impressions", "clicks", "conversions"]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def analyze_campaigns(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Clean campaign records, calculate KPIs, and flag conversion outliers."""
    frame = pd.DataFrame(records)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("At least one campaign row is required")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["campaign"] = frame["campaign"].fillna("").astype(str).str.strip()
    frame.loc[frame["campaign"].eq(""), "campaign"] = "Unlabelled campaign"
    for column in METRIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[METRIC_COLUMNS].isna().any().any():
        raise ValueError("Metric values must be valid numbers")
    if (frame[METRIC_COLUMNS] < 0).any().any():
        raise ValueError("Metric values cannot be negative")

    rows_received = len(frame)
    frame = frame.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first").reset_index(drop=True)
    totals = {column: round(float(frame[column].sum()), 2) for column in METRIC_COLUMNS}
    kpis = {
        "ctr": _safe_ratio(totals["clicks"], totals["impressions"]),
        "conversion_rate": _safe_ratio(totals["conversions"], totals["clicks"]),
        "cpc": _safe_ratio(totals["spend"], totals["clicks"]),
        "cpa": _safe_ratio(totals["spend"], totals["conversions"]),
    }

    method = "iqr"
    if len(frame) >= 8 and IsolationForest is not None:
        model = IsolationForest(contamination="auto", random_state=42)
        labels = model.fit_predict(frame[["conversions"]])
        anomaly_mask = labels == -1
        method = "isolation_forest"
    else:
        values = frame["conversions"]
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        anomaly_mask = (values < lower) | (values > upper)

    details = []
    for idx, row in frame.iterrows():
        details.append({
            "campaign": row["campaign"],
            "spend": round(float(row["spend"]), 2),
            "impressions": int(row["impressions"]),
            "clicks": int(row["clicks"]),
            "conversions": int(row["conversions"]),
            "ctr": _safe_ratio(float(row["clicks"]), float(row["impressions"])),
            "conversion_rate": _safe_ratio(float(row["conversions"]), float(row["clicks"])),
            "cpa": _safe_ratio(float(row["spend"]), float(row["conversions"])),
            "is_conversion_outlier": bool(anomaly_mask.iloc[idx]),
        })

    suggestions = []
    if totals["spend"] > 0 and totals["conversions"] == 0:
        suggestions.append("Spend is recorded but no conversions are present; verify conversion tracking before reallocating budget.")
    if totals["impressions"] > 0 and totals["clicks"] == 0:
        suggestions.append("Impressions are recorded without clicks; review audience and creative relevance.")
    outliers = [item for item in details if item["is_conversion_outlier"]]
    if outliers:
        names = ", ".join(item["campaign"] for item in outliers)
        suggestions.append(f"Review conversion-count outlier(s) {names}; compare tracking and campaign context before acting.")
    if not suggestions:
        suggestions.append("No basic tracking or conversion outlier signals were found in this sample.")

    return {
        "rows_received": rows_received,
        "rows_analyzed": len(frame),
        "duplicate_rows_removed": rows_received - len(frame),
        "totals": totals,
        "kpis": kpis,
        "anomaly_method": method,
        "campaigns": details,
        "suggestions": suggestions,
    }
