"""Tests for hexabin/edge_client.py — classify_remote (edge → server RPC)."""

import base64
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from hexabin.edge_client import EdgeServerError, classify_remote


def _ok_response(body) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode()
    return resp


class TestClassifyRemote:
    def test_missing_server_url_raises(self):
        with patch("hexabin.edge_client.SERVER_URL", ""):
            with pytest.raises(EdgeServerError, match="SERVER_URL"):
                classify_remote(b"img")

    def test_success_returns_parsed_body(self):
        body = {"status": "ok", "id": 1, "result": {"category": "Glass"}, "command": {}}
        with (
            patch("hexabin.edge_client.SERVER_URL", "http://server:8000"),
            patch("hexabin.edge_client.urllib.request.urlopen") as mock_open,
        ):
            mock_open.return_value.__enter__.return_value = _ok_response(body)
            assert classify_remote(b"img") == body

    def test_payload_and_url(self):
        env = {"simulated_temperature": 21.0, "simulated_smoke": 0.3}
        with (
            patch("hexabin.edge_client.SERVER_URL", "http://server:8000/"),
            patch("hexabin.edge_client.urllib.request.urlopen") as mock_open,
        ):
            mock_open.return_value.__enter__.return_value = _ok_response({"status": "ok"})
            classify_remote(b"img-bytes", env=env)
            req = mock_open.call_args[0][0]

        assert req.full_url == "http://server:8000/api/edge/classify"
        payload = json.loads(req.data)
        assert payload["image_b64"] == base64.b64encode(b"img-bytes").decode()
        assert payload["bin_id"]
        assert payload["captured_at"]
        assert payload["simulated_temperature"] == 21.0
        assert payload["simulated_smoke"] == 0.3
        assert req.get_header("Authorization", "").startswith("Bearer ")

    def test_http_error_raises_with_status_and_detail(self):
        err = urllib.error.HTTPError(
            "http://server:8000/api/edge/classify",
            502,
            "bad gateway",
            {},
            io.BytesIO(b'{"error": "classification failed"}'),
        )
        with (
            patch("hexabin.edge_client.SERVER_URL", "http://server:8000"),
            patch("hexabin.edge_client.urllib.request.urlopen", side_effect=err),
        ):
            with pytest.raises(EdgeServerError, match="502"):
                classify_remote(b"img")

    def test_network_error_raises(self):
        with (
            patch("hexabin.edge_client.SERVER_URL", "http://server:8000"),
            patch(
                "hexabin.edge_client.urllib.request.urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            ),
        ):
            with pytest.raises(EdgeServerError):
                classify_remote(b"img")

    def test_timeout_raises(self):
        with (
            patch("hexabin.edge_client.SERVER_URL", "http://server:8000"),
            patch(
                "hexabin.edge_client.urllib.request.urlopen",
                side_effect=TimeoutError("timed out"),
            ),
        ):
            with pytest.raises(EdgeServerError):
                classify_remote(b"img")

    def test_non_dict_response_raises(self):
        with (
            patch("hexabin.edge_client.SERVER_URL", "http://server:8000"),
            patch("hexabin.edge_client.urllib.request.urlopen") as mock_open,
        ):
            mock_open.return_value.__enter__.return_value = _ok_response(["not", "a", "dict"])
            with pytest.raises(EdgeServerError):
                classify_remote(b"img")
