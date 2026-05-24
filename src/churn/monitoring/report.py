# flake8: noqa: E501
"""CLI: generate an HTML drift report.

Usage::

    python -m churn.monitoring.report \
        --reference data/processed/reference.parquet \
        --current   data/incoming/last_week.parquet \
        --output    reports/drift_2026-01-22.html

The HTML page contains a single table: one row per feature with PSI and
severity, colour-coded against the thresholds in :mod:`churn.config`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from churn import config
from churn.monitoring.drift import feature_drift_report

logger = logging.getLogger(__name__)


_SEVERITY_COLOUR = {
    "none": "#d4edda",  # green
    "moderate": "#fff3cd",  # yellow
    "significant": "#f8d7da",  # red
    "unknown": "#e2e3e5",  # grey
}


def render_html(
    drift_df: pd.DataFrame, reference_path: Path, current_path: Path
) -> str:
    """Render the drift summary as an HTML page."""
    rows = []
    for _, row in drift_df.iterrows():
        colour = _SEVERITY_COLOUR.get(row["severity"], "#ffffff")
        rows.append(
            f'<tr style="background-color: {colour};">'
            f'<td>{row["feature"]}</td>'
            f'<td style="text-align: right;">{row["psi"]:.4f}</td>'
            f'<td>{row["severity"]}</td>'
            f"</tr>"
        )
    table_rows = (
        "\n".join(rows)
        if rows
        else ('<tr><td colspan="3">No features compared.</td></tr>')
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>RetailGenius — Churn drift report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222; }}
        h1 {{ margin-bottom: 0.25rem; }}
        .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
        th, td {{ padding: 0.5rem 0.75rem; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .legend {{ margin-top: 1.5rem; font-size: 0.9rem; }}
        .legend span {{ display: inline-block; padding: 0.25rem 0.6rem; margin-right: 0.5rem;
                        border: 1px solid #ccc; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Churn — feature drift report</h1>
    <div class="meta">
        Generated: {datetime.now().isoformat(timespec="seconds")}<br/>
        Reference: <code>{reference_path}</code><br/>
        Current:   <code>{current_path}</code>
    </div>

    <table>
        <thead>
            <tr><th>Feature</th><th>PSI</th><th>Severity</th></tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <div class="legend">
        <span style="background: {_SEVERITY_COLOUR['none']};">
            PSI &lt; {config.PSI_NO_SHIFT} — none
        </span>
        <span style="background: {_SEVERITY_COLOUR['moderate']};">
            {config.PSI_NO_SHIFT} ≤ PSI &lt; {config.PSI_MODERATE_SHIFT} — moderate
        </span>
        <span style="background: {_SEVERITY_COLOUR['significant']};">
            PSI ≥ {config.PSI_MODERATE_SHIFT} — significant
        </span>
    </div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=config.REFERENCE_FILE,
        help="Parquet file containing the reference distribution.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Parquet file containing the current sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.REPORTS_DIR / "drift_report.html",
        help="Where to write the HTML report.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")
    if not args.current.exists():
        raise FileNotFoundError(f"Current file not found: {args.current}")

    reference_df = pd.read_parquet(args.reference)
    current_df = pd.read_parquet(args.current)

    drift_df = feature_drift_report(reference_df, current_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(drift_df, args.reference, args.current))
    logger.info("Wrote drift report to %s", args.output)

    # Also print a short summary to stdout for use in shell pipelines
    if not drift_df.empty:
        print(drift_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
