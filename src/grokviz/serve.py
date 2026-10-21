"""Local preview server for the built site, with caching disabled (`python -m grokviz.serve`).

Local explanation: Python's built-in static file server, plus a `Cache-Control: no-store` header so the
browser always fetches the latest JavaScript modules and data after a rebuild (ES modules are otherwise cached
by heuristic freshness, which hides changes during development).
Global context: used by .claude/launch.json and README's "preview locally" instructions; the deployed site
is served by GitHub Pages instead.
Source: http.server (https://docs.python.org/3/library/http.server.html).
"""

from __future__ import annotations  # postponed annotations

import argparse  # --dir / --port
import functools  # bind the directory to the handler
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer  # static files, one thread per request


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Static file handler that forbids caching."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always revalidate: development only
        super().end_headers()


def main() -> None:
    """Serve --dir on --port until interrupted."""
    parser = argparse.ArgumentParser(description="Serve the built site without caching")
    parser.add_argument("--dir", default="site")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = functools.partial(NoCacheHandler, directory=args.dir)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"serving {args.dir} at http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
