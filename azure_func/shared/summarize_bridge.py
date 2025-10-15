"""Bridge helpers for invoking :mod:`summarize` from the Azure Function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

import summarize
from .logging_utils import get_json_logger

LOGGER = get_json_logger("mmqb.summarize_bridge")

@dataclass
class SummaryArtifacts:
    """Outputs produced by the summarization pipeline."""
    csv_path: str | None
    summary_path: str | None
    summary: Dict[str, Any]

def _records_to_dataframe(
    records: Sequence[Dict[str, Any]],
    dataset_name: str,
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    """Validate and prepare records for summarization."""
    LOGGER.info("Dataset received", extra={"event": "dataset_received", "dataset": dataset_name, "row_count": len(records)})
    if not records:
        raise ValueError(f"{dataset_name.capitalize()} dataset is empty")

    df = pd.DataFrame.from_records(list(records))
    LOGGER.info(
        "DataFrame constructed",
        extra={"event": "df_constructed", "dataset": dataset_name, "shape": list(df.shape), "columns": list(df.columns)[:50]},
    )
    if df.empty:
        raise ValueError(f"{dataset_name.capitalize()} dataset produced an empty DataFrame")

    try:
        summarize.ensure_columns(df)
    except ValueError as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Missing required columns",
            extra={"event": "missing_required_columns", "dataset": dataset_name, "error": str(exc)},
        )
        raise ValueError(f"{dataset_name.capitalize()} dataset {exc}") from exc

    before_rows = len(df)
    df = summarize.prepare_df(df)
    after_rows = len(df)
    LOGGER.info(
        "Data prepared",
        extra={
            "event": "df_prepared",
            "dataset": dataset_name,
            "rows_before": before_rows,
            "rows_after": after_rows,
            "has_WEEK_ENDING": "WEEK_ENDING" in df.columns,
        },
    )
    if df.empty:
        raise ValueError(f"{dataset_name.capitalize()} dataset has no usable rows after preparation")

    week_values = sorted(pd.to_datetime(df["WEEK_ENDING"].dropna()).unique().tolist())
    LOGGER.info(
        "Week-ending analysis",
        extra={
            "event": "week_stats",
            "dataset": dataset_name,
            "distinct_weeks": len(week_values),
            "min_week": str(week_values[0]) if week_values else None,
            "max_week": str(week_values[-1]) if week_values else None,
        },
    )
    if not week_values:
        raise ValueError(f"{dataset_name.capitalize()} dataset has no valid DELIVERY_DT values")

    return df, [pd.Timestamp(w) for w in week_values]

def _derive_label(weeks: Iterable[pd.Timestamp]) -> str:
    return summarize.format_label(list(weeks))

def generate_summary_artifacts(
    current_records: Sequence[Dict[str, Any]],
    *,
    operation_cd: str,
    label_current: str | None = None,
    label_baseline: str | None = None,
) -> SummaryArtifacts:
    """Compute the MMQB summary from a single dataset of JSON record lists."""
    op_code = (operation_cd or "").strip()
    if not op_code:
        raise ValueError("operation_cd must be supplied")

    df_all, weeks_all = _records_to_dataframe(current_records, "current")

    current_weeks, baseline_weeks = summarize.pick_periods_by_weeks(weeks_all)
    LOGGER.info(
        "Period selection complete",
        extra={
            "event": "periods_chosen",
            "weeks_all": len(weeks_all),
            "weeks_A": len(current_weeks or []),
            "weeks_B": len(baseline_weeks or []),
            "A_last": str(current_weeks[-1]) if current_weeks else None,
            "B_last": str(baseline_weeks[-1]) if baseline_weeks else None,
        },
    )
    if not current_weeks or not baseline_weeks:
        raise ValueError("Current dataset must contain enough distinct weeks to derive comparison periods")

    df_current = df_all[df_all["WEEK_ENDING"].isin(current_weeks)].copy()
    df_baseline = df_all[df_all["WEEK_ENDING"].isin(baseline_weeks)].copy()
    LOGGER.info(
        "Period dataframes sliced",
        extra={
            "event": "periods_sliced",
            "rows_A": len(df_current),
            "rows_B": len(df_baseline),
        },
    )

    current_label = label_current or _derive_label(current_weeks)
    baseline_label = label_baseline or _derive_label(baseline_weeks)
    LOGGER.info(
        "Labels derived",
        extra={"event": "labels_ready", "label_A": current_label, "label_B": baseline_label},
    )

    summary = summarize.compute_summary(
        df_current,
        df_baseline,
        op_code,
        current_label,
        baseline_label,
    )
    LOGGER.info("Summary computed", extra={"event": "summary_done", "top10_count": len(summary.get("top10", []))})
    return SummaryArtifacts(csv_path=None, summary_path=None, summary=summary)
