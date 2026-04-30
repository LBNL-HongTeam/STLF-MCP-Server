#!/usr/bin/env python3
"""
LoadForecasting-MCP Server Entry Point

Supports dual transport modes:
- STDIO (default): For Claude Desktop and direct MCP clients
- HTTP: For AlphaBuilding-Agents and web integrations

Usage:
    # STDIO mode (default)
    python main.py

    # HTTP mode
    python main.py --http
    # or
    MCP_TRANSPORT=http python main.py
"""

import argparse
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load_forecasting.server import mcp, run_server


def main():
    parser = argparse.ArgumentParser(
        description="LoadForecasting MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run in STDIO mode (Claude Desktop)
    python main.py --http       # Run in HTTP mode (port 8003)
    python main.py --http 8080  # Run in HTTP mode on custom port
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

    # Determine transport mode
    transport_mode = os.getenv("MCP_TRANSPORT", "stdio").lower()

    if args.http is not None:
        transport_mode = "http"
        if isinstance(args.http, str) and args.http.isdigit():
            os.environ["MCP_HTTP_PORT"] = args.http

    run_server(transport_mode)


if __name__ == "__main__":
    main()
