"""
Assemble the final self-contained HTML report.

Reads the Jinja template, the CSS, the vendored JS libraries and the
chart-rendering JS from disk and inlines them into a single string.
"""

from __future__ import annotations

from ._render import read_asset, render_report
from .payload import ReportPayload, payload_to_json


def build_report_html(payload: ReportPayload) -> str:
    """
    Render the payload to a single self-contained HTML document.

    Args:
        payload: ReportPayload returned by build_report_payload.

    Returns:
        Complete HTML document as a string.  Every external dependency
        (CSS, d3, Observable Plot, chart JS, data) is inlined, so the
        result works when opened via file:// in any modern browser.
    """
    title = (payload.meta or {}).get("title") or "Evaluation report"
    return render_report(
        "template.html",
        title=title,
        charts_js=read_asset("charts.js"),
        payload_json=payload_to_json(payload),
    )
