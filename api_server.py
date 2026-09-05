"""Serve the Ingrid HTML UI and a local Pinecone-backed product API."""

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "ingredient-checker"))

from chain import analyse_label


class IngridHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if urlparse(self.path).path == "/ingrid_your_ingredient_decoded.html":
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", "/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/product":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            barcode = str(payload.get("barcode", "")).strip()
            result = analyse_label(barcode)
        except (ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON request")
            return

        body = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("", 8000), IngridHandler)
    print("Ingrid is running at http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()