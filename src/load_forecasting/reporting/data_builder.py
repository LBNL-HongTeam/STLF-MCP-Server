"""
Assemble the self-contained HTML data-inspection report.

Same recipe as the evaluation / backtest / inference reports: the payload is
serialised to JSON and inlined, together with the CSS, the vendored d3 and
Observable Plot bundles, and the chart JS, so the file opens via file://
with no network access.
"""

from __future__ import annotations

from ._render import read_asset, render_report
from .data_payload import DataReportPayload, payload_to_json


def build_data_report_html(payload: DataReportPayload) -> str:
    title = (payload.meta or {}).get("title") or "Data report"
    return render_report(
        "data_template.html",
        title=title,
        charts_js=read_asset("data_charts.js"),
        payload_json=payload_to_json(payload),
    )
