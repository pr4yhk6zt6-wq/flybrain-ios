#!/usr/bin/env python3
"""
Serve the world.

    python3 tools/serve_world.py [--port 8000] [--dir world]

A plain static file server over the recording `tools/step4_world.py` wrote.
It binds 0.0.0.0 and does not care what Host header it is reached by, so it
works behind a proxy as happily as on a laptop. There is nothing to install:
three.js is vendored in `world/` and the recording is a pair of binary files.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import pathlib
import socketserver


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".bin": "application/octet-stream",
        ".json": "application/json",
        ".js": "text/javascript",
    }

    def end_headers(self):
        # the files are large and never change while a run is being watched
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="world", type=pathlib.Path)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist — run tools/step4_world.py first")

    handler = functools.partial(Handler, directory=str(root))
    with Server((args.bind, args.port), handler) as httpd:
        print(f"the world is at http://{args.bind}:{args.port}/  ({root})")
        print("ctrl-C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
