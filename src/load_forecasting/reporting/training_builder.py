"""
Assemble the self-contained HTML training report (learning curves, metrics,
splits, tuning study, compute) for one or more trained models.
"""

from __future__ import annotations

from ._render import read_asset, render_report
from .training_payload import TrainingReportPayload, payload_to_json


def build_training_report_html(payload: TrainingReportPayload) -> str:
    title = (payload.meta or {}).get("title") or "Training report"
    return render_report(
        "training_template.html",
        title=title,
        charts_js=read_asset("training_charts.js"),
        payload_json=payload_to_json(payload),
    )
