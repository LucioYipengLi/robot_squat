"""Exercise viewer response routes without opening a listening socket."""

import gzip
import io
import json
from email.message import Message

import numpy as np
import pytest

from g1.rl.failures.storage import FailureArchive, write_json
from g1.rl.failures.viewer import ArchiveHandler


@pytest.fixture
def archive(tmp_path):
    result = FailureArchive(tmp_path / "run", {"partners": ["ground"]})
    frames = {
        "time": np.array([0.1]),
        "body_pose": np.array([[[0, 0, 0.75, 0, 0, 0, 1]]]),
        "goal_pose": np.zeros((1, 2, 7)),
        "pair_force": np.zeros((1, 2, 1, 3)),
        "net_force": np.zeros((1, 2, 3)),
        "contact_pos": np.full((1, 1, 3), np.nan),
    }
    result.add_episode({"failed": True, "category": "planar", "elapsed_s": 0.1, "reasons": ["base_contact"]}, frames)
    result.close()
    write_json(result.directory / "geometry.json", {"body_names": ["torso_link"], "meshes": [], "edges": []})
    return result.directory


def response(directory, path, *, compressed=False):
    handler = ArchiveHandler.__new__(ArchiveHandler)
    handler.directory, handler.path = directory, path
    handler.headers = Message()
    if compressed:
        handler.headers["Accept-Encoding"] = "gzip"
    handler.wfile = io.BytesIO()
    result = {"headers": {}}
    handler.send_response = lambda code: result.update(status=code)
    handler.send_header = lambda key, value: result["headers"].update({key: value})
    handler.end_headers = lambda: None
    handler.send_error = lambda code, *args: result.update(status=code)
    handler.do_GET()
    result["body"] = handler.wfile.getvalue()
    return result


@pytest.mark.parametrize("route", ["/api/index", "/api/geometry", "/api/sample/000000"])
def test_json_routes_are_valid_and_read_only(archive, route):
    result = response(archive, route)
    assert result["status"] == 200
    assert result["headers"]["Cache-Control"] == "no-store"
    data = json.loads(result["body"])
    if route.startswith("/api/sample"):
        assert data["contact_pos"] == [[[None, None, None]]]


def test_html_can_be_gzip_compressed_and_has_no_external_assets(archive):
    result = response(archive, "/", compressed=True)
    assert result["status"] == 200 and result["headers"]["Content-Encoding"] == "gzip"
    html = gzip.decompress(result["body"]).decode()
    assert "webgl2" in html and "<script>" in html
    assert 'src="http' not in html and 'href="http' not in html


@pytest.mark.parametrize("route", ["/../../.git/config", "/api/sample/../../run.json", "/api/sample/000099"])
def test_unlisted_paths_and_missing_samples_are_not_served(archive, route):
    result = response(archive, route)
    assert result["status"] in (400, 404)
