"""
Assemble the final self-contained HTML backtest report.

Mirrors builder.py but uses the backtest template and backtest chart JS.
"""

from __future__ import annotations

from ._render import read_asset, render_report
from .backtest_payload import BacktestReportPayload, backtest_payload_to_json


def build_backtest_report_html(payload: BacktestReportPayload) -> str:
    """
    Render a BacktestReportPayload to a single self-contained HTML document.

    Every external dependency (CSS, d3, Observable Plot, chart JS, data) is
    inlined so the result works when opened via file:// in any modern browser.

    Args:
        payload: BacktestReportPayload returned by build_backtest_payload.

    Returns:
        Complete HTML document as a string.
    """
    title = (payload.meta or {}).get("title") or "Backtest report"
    return render_report(
        "backtest_template.html",
        title=title,
        charts_js=read_asset("backtest_charts.js"),
        payload_json=backtest_payload_to_json(payload),
    )
