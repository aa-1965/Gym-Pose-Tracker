#!/usr/bin/env python3
"""Serve the GymForm UI only (no pose backend). Use api_server.py for full integration."""

import http.server
import os
import webbrowser
from pathlib import Path

PORT = 8080
UI_DIR = Path(__file__).parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)


def main():
    os.chdir(UI_DIR)
    url = f"http://localhost:{PORT}"
    print(f"GymForm UI only at {url}")
    print("For pose tracking, run: python api_server.py")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    http.server.HTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
