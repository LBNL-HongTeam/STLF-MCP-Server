"""
Shared rendering utilities for the reporting package.

Consolidates static-asset reading, HTML-safe JSON serialisation, and the
common Jinja render path used by the evaluation, backtest, and inference
report builders.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Template

_HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def read_asset(name: str) -> str:
    """Read a static asset file shipped with the reporting package."""
    path = _HERE / name
    if not path.exists():
        raise FileNotFoundError(f"Reporting asset not found: {name}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_template(name: str) -> Template:
    """Load and cache a Jinja template by filename.

    autoescape is intentionally off — every injected value is either trusted
    JS/CSS sourced from disk or pre-escaped JSON.
    """
    src = (_HERE / name).read_text(encoding="utf-8")
    return Template(src, autoescape=False)


def scrub_nonfinite(obj: Any) -> Any:
    """Recursively replace NaN / Inf and numpy scalars with JSON-safe values."""
    if isinstance(obj, dict):
        return {k: scrub_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_nonfinite(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    # numpy scalars/arrays handled lazily to avoid a hard numpy dependency here
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        return scrub_nonfinite(obj.tolist())
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        try:
            return scrub_nonfinite(obj.item())
        except Exception:
            return obj
    return obj


def to_html_json(obj: Any, *, encoder: type[json.JSONEncoder] | None = None) -> str:
    """Serialise an object to a JSON string safe to embed in an HTML <script>.

    Non-finite floats are scrubbed to null, and sequences that could
    terminate the surrounding script block are escaped.
    """
    scrubbed = scrub_nonfinite(obj)
    if encoder is not None:
        raw = json.dumps(scrubbed, cls=encoder, ensure_ascii=False, allow_nan=False)
    else:
        raw = json.dumps(scrubbed, default=str)
    return (
        raw.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_report(
    template_name: str,
    *,
    title: str,
    charts_js: str,
    payload_json: str,
    **extra: Any,
) -> str:
    """Render a report template, inlining the shared CSS + vendored JS libs."""
    template = load_template(template_name)
    return template.render(
        title=title,
        inlined_css=read_asset("styles.css"),
        d3_js=read_asset("assets/d3.min.js"),
        plot_js=read_asset("assets/observable-plot.min.js"),
        charts_js=charts_js,
        payload_json=payload_json,
        **extra,
    )
