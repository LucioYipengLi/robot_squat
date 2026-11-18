"""Read-only localhost viewer; run archives contain arrays and JSON, never pickled Python."""

import gzip
import json
import re
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .storage import plain


def read_index(directory):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run.get("schema_version") != 1:
        raise ValueError("Unsupported failure archive schema")
    samples = []
    with (directory / "failures.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.endswith("\n"):
                samples.append(json.loads(line))
    return {
        "run": run,
        "summary": json.loads((directory / "summary.json").read_text(encoding="utf-8")),
        "samples": samples,
    }


def sample_payload(directory, sample_id):
    if not re.fullmatch(r"\d{6}", sample_id):
        raise ValueError("Invalid sample id")
    directory = Path(directory).resolve()
    path = (directory / "clips" / f"{sample_id}.npz").resolve()
    if path.parent != directory / "clips":
        raise ValueError("Clip must be inside the archive")
    with np.load(path, allow_pickle=False) as data:
        names = ("time", "body_pose", "goal_pose", "pair_force", "net_force", "contact_pos")
        return plain({key: np.asarray(data[key], dtype=np.float64).round(6) for key in names})


class ArchiveHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory, **kwargs):
        self.directory = Path(directory)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        try:
            path = urlsplit(self.path).path
            if path == "/":
                content = Path(__file__).with_name("viewer.html").read_bytes()
                kind = "text/html; charset=utf-8"
            else:
                kind = "application/json; charset=utf-8"
                if path == "/api/index":
                    value = read_index(self.directory)
                elif path == "/api/geometry":
                    value = json.loads((self.directory / "geometry.json").read_text(encoding="utf-8"))
                elif path.startswith("/api/sample/"):
                    value = sample_payload(self.directory, path.removeprefix("/api/sample/"))
                else:
                    self.send_error(404)
                    return
                content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            compressed = "gzip" in self.headers.get("Accept-Encoding", "") and len(content) > 4096
            if compressed:
                content = gzip.compress(content)
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(content)
        except (OSError, ValueError, KeyError) as error:
            self.send_error(400, "Invalid or incomplete failure archive", str(error))

    def log_message(self, *_):
        pass


def serve(directory, *, port=8765, open_browser=True):
    directory = Path(directory).resolve()
    read_index(directory)
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(ArchiveHandler, directory=directory))
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"[failures] Read-only viewer: {address}\n[failures] Archive: {directory}", flush=True)
    try:
        if open_browser:
            webbrowser.open(address)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
