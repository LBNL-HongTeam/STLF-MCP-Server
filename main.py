#!/usr/bin/env python3
"""
LoadForecasting-MCP Server Entry Point

Supports dual transport modes:
- STDIO (default): For Claude Desktop and direct MCP clients
- HTTP: For AlphaBuilding-Agents and web integrations

Usage:
    python main.py              # STDIO mode
    python main.py --http       # HTTP mode on default port 8003
    python main.py --http 8080  # HTTP mode on port 8080
    MCP_TRANSPORT=http python main.py
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# OpenMP duplicate-runtime guard — MUST run before torch / xgboost are imported.
#
# On macOS, xgboost's libxgboost.dylib links @rpath/libomp.dylib which resolves
# to the Homebrew copy (/opt/homebrew/.../libomp.dylib), while PyTorch/scipy load
# the venv's own libomp.dylib. Two OpenMP runtimes in one process is undefined
# behaviour: once both are resident (e.g. after the long-lived server has handled
# an XGBoost call and later runs a Torch model's historical_forecasts), the
# conflicting thread barriers can hard-crash the process with a SIGSEGV in an
# libomp worker thread (EXC_BAD_ACCESS in __kmp_hyper_barrier_release).
#
# KMP_DUPLICATE_LIB_OK=TRUE tells the OpenMP runtime to tolerate the duplicate
# instead of aborting. OMP_NUM_THREADS caps barrier contention. Both are only
# applied if the user has not already set them, so shell/.env overrides win.
#
# See AGENTS.md ("OpenMP duplicate-runtime crash") for the full diagnosis.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Add src to path for development (no-op when installed via pip)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load_forecasting.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoadForecasting MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # STDIO mode (Claude Desktop)
    python main.py --http       # HTTP mode (port 8003)
    python main.py --http 8080  # HTTP mode on custom port
        """,
    )
    parser.add_argument(
        "--http",
        nargs="?",
        const=True,
        default=None,
        help="Run in HTTP mode. Optionally specify port (default: 8003)",
    )
    args = parser.parse_args()

    # CLI flag takes precedence over env var
    if args.http is not None:
        transport_mode = "http"
        port = int(args.http) if isinstance(args.http, str) and args.http.isdigit() else 8003
    else:
        transport_mode = os.getenv("MCP_TRANSPORT", "stdio").lower()
        port = int(os.getenv("MCP_HTTP_PORT", "8003"))

    run_server(transport_mode, port=port)


if __name__ == "__main__":
    main()
