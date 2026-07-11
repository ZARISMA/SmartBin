"""Tests for smartwaste/web.py — FastAPI endpoints."""

import base64
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from smartwaste.config import ADMIN_PASSWORD, ADMIN_USERNAME
from smartwaste.llm import CircuitOpenError, ClassificationResult, LLMError

# Matches SMARTWASTE_EDGE_API_KEY set in conftest.py before smartwaste imports.
EDGE_HEADERS = {"Authorization": "Bearer test-edge-key"}


@pytest.fixture
def client():
    # Prevent camera thread from starting during tests
    with patch("smartwaste.web._start_camera_thread"):
        from smartwaste.web import app

        with TestClient(app) as c:
            # Authenticate so protected routes return real responses
            c.post(
                "/login",
                data={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
                follow_redirects=False,
            )
            yield c


@pytest.fixture
def anon_client():
    """TestClient WITHOUT a session login — for bearer-token auth tests."""
    with patch("smartwaste.web._start_camera_thread"):
        from smartwaste.web import app

        with TestClient(app) as c:
            yield c


class TestIndex:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_contains_title(self, client):
        r = client.get("/")
        assert "SmartBin" in r.text and "Fleet Control" in r.text


class TestApiState:
    def test_returns_json(self, client):
        r = client.get("/api/state")
        assert r.status_code == 200
        data = r.json()
        assert "label" in data
        assert "detail" in data
        assert "auto_on" in data
        assert "history" in data

    def test_initial_label_is_ready(self, client):
        r = client.get("/api/state")
        assert r.json()["label"] == "Ready"


class TestApiToggleAuto:
    def test_toggles_auto(self, client):
        r = client.post("/api/toggle-auto")
        assert r.status_code == 200
        assert "auto_classify" in r.json()


class TestApiClassify:
    def test_no_frame_returns_503(self, client):
        r = client.post("/api/classify")
        assert r.status_code == 503


class TestApiEntries:
    def test_returns_list(self, client):
        r = client.get("/api/entries")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestApiStats:
    def test_returns_stats(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "by_category" in data


class TestApiAnalytics:
    def test_returns_full_payload(self, client):
        r = client.get("/api/analytics?period=7d")
        assert r.status_code == 200
        data = r.json()
        for key in (
            "period",
            "period_label",
            "granularity",
            "range",
            "kpis",
            "series",
            "by_category",
            "by_category_prev",
            "leaderboard",
            "backends",
        ):
            assert key in data
        assert data["period"] == "7d"
        assert set(data["kpis"]) == {"total", "diversion_rate", "avg_confidence", "active_bins"}
        assert len(data["series"]["buckets"]) == 7

    def test_hourly_granularity_for_24h(self, client):
        data = client.get("/api/analytics?period=24h").json()
        assert data["granularity"] == "hour"
        assert len(data["series"]["buckets"]) == 24

    def test_invalid_period_returns_400(self, client):
        assert client.get("/api/analytics?period=bogus").status_code == 400

    def test_anon_rejected(self, anon_client):
        assert anon_client.get("/api/analytics").status_code == 401

    def test_edge_key_rejected(self, anon_client):
        assert anon_client.get("/api/analytics", headers=EDGE_HEADERS).status_code == 401


class TestApiAnalyticsExport:
    def test_returns_csv_attachment(self, client):
        from smartwaste import analytics

        r = client.get("/api/analytics/export?period=24h")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert "smartbin-classifications-24h.csv" in r.headers["content-disposition"]
        assert r.text.splitlines()[0] == ",".join(analytics.EXPORT_COLUMNS)

    def test_invalid_period_returns_400(self, client):
        assert client.get("/api/analytics/export?period=nope").status_code == 400

    def test_anon_rejected(self, anon_client):
        assert anon_client.get("/api/analytics/export").status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Auth scoping — the edge key only opens the ingest endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthScoping:
    def test_edge_key_rejected_on_admin_api(self, anon_client):
        assert anon_client.get("/api/entries", headers=EDGE_HEADERS).status_code == 401

    def test_edge_key_rejected_on_dashboard_api(self, anon_client):
        assert anon_client.get("/api/dashboard", headers=EDGE_HEADERS).status_code == 401

    def test_admin_bearer_allowed_on_admin_api(self, anon_client):
        r = anon_client.get("/api/entries", headers={"Authorization": f"Bearer {ADMIN_PASSWORD}"})
        assert r.status_code == 200

    def test_edge_key_allowed_on_heartbeat(self, anon_client):
        r = anon_client.post(
            "/api/heartbeat", json={"bin_id": "bin-authtest"}, headers=EDGE_HEADERS
        )
        assert r.status_code == 200

    def test_no_auth_rejected_on_edge_classify(self, anon_client):
        r = anon_client.post("/api/edge/classify", json=_classify_payload())
        assert r.status_code == 401

    def test_no_auth_rejected_on_report(self, anon_client):
        r = anon_client.post(
            "/api/report",
            json={"bin_id": "b", "label": "Plastic", "timestamp": "2026-07-09 12:00:00"},
        )
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/edge/classify — server-side LLM classification
# ─────────────────────────────────────────────────────────────────────────────


def _llm_result(category="Plastic", confidence=0.92):
    return ClassificationResult(
        category=category,
        description="a bottle",
        brand_product="Jermuk",
        confidence=confidence,
        backend="lmstudio",
    )


def _classify_payload(image: bytes = b"fake-jpeg-bytes") -> dict:
    return {
        "bin_id": "bin-test",
        "image_b64": base64.b64encode(image).decode(),
        "captured_at": "2026-07-09 12:00:00",
        "location": "Yerevan",
    }


class TestEdgeClassify:
    def _post(self, client, tmp_path, *, payload=None, result=None, error=None, insert_id=42):
        backend = MagicMock()
        if error is not None:
            backend.classify.side_effect = error
        else:
            backend.classify.return_value = result or _llm_result()
        with (
            patch("smartwaste.web.build_backend", return_value=backend),
            patch("smartwaste.web.DATASET_DIR", str(tmp_path)),
            patch("smartwaste.web.insert_entry", return_value=insert_id) as mock_insert,
        ):
            r = client.post(
                "/api/edge/classify", json=payload or _classify_payload(), headers=EDGE_HEADERS
            )
        return r, mock_insert

    def test_happy_path_returns_result_and_command(self, anon_client, tmp_path):
        r, mock_insert = self._post(anon_client, tmp_path)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["id"] == 42
        assert data["result"]["category"] == "Plastic"
        assert data["result"]["backend"] == "lmstudio"
        assert data["command"] == {"action": "open_module", "module": 1, "category": "Plastic"}
        assert mock_insert.called

    def test_entry_carries_confidence_backend_and_bin(self, anon_client, tmp_path):
        _, mock_insert = self._post(anon_client, tmp_path)
        entry = mock_insert.call_args[0][0]
        assert entry["confidence"] == pytest.approx(0.92)
        assert entry["llm_backend"] == "lmstudio"
        assert entry["bin_id"] == "bin-test"
        assert entry["timestamp"] == "2026-07-09 12:00:00"

    def test_image_saved_into_dataset_dir(self, anon_client, tmp_path):
        self._post(anon_client, tmp_path)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".jpg"

    def test_empty_returns_none_command_without_insert(self, anon_client, tmp_path):
        r, mock_insert = self._post(anon_client, tmp_path, result=_llm_result("Empty", 0.99))
        data = r.json()
        assert data["status"] == "ok"
        assert data["id"] is None
        assert data["command"]["action"] == "none"
        assert data["command"]["module"] is None
        assert not mock_insert.called

    def test_db_failure_still_returns_command(self, anon_client, tmp_path):
        r, _ = self._post(anon_client, tmp_path, insert_id=None)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "db_error"
        assert data["command"]["action"] == "open_module"

    def test_invalid_base64_returns_400(self, anon_client, tmp_path):
        payload = _classify_payload()
        payload["image_b64"] = "!!!not-base64!!!"
        r, _ = self._post(anon_client, tmp_path, payload=payload)
        assert r.status_code == 400

    def test_oversized_image_returns_413(self, anon_client, tmp_path):
        with patch("smartwaste.web.MAX_UPLOAD_BYTES", 4):
            r, _ = self._post(anon_client, tmp_path)
        assert r.status_code == 413

    def test_backend_error_returns_502(self, anon_client, tmp_path):
        r, _ = self._post(anon_client, tmp_path, error=LLMError("LM Studio down"))
        assert r.status_code == 502

    def test_circuit_open_returns_503(self, anon_client, tmp_path):
        r, _ = self._post(anon_client, tmp_path, error=CircuitOpenError("paused"))
        assert r.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/report — hardened persistence endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestApiReport:
    def _payload(self, **overrides):
        p = {"bin_id": "bin-9", "label": "Plastic", "timestamp": "2026-07-09 12:00:00"}
        p.update(overrides)
        return p

    def test_insert_success_returns_id(self, anon_client):
        with patch("smartwaste.web.insert_entry", return_value=7):
            r = anon_client.post("/api/report", json=self._payload(), headers=EDGE_HEADERS)
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "id": 7}

    def test_insert_failure_returns_500(self, anon_client):
        with patch("smartwaste.web.insert_entry", return_value=None):
            r = anon_client.post("/api/report", json=self._payload(), headers=EDGE_HEADERS)
        assert r.status_code == 500

    def test_traversal_fields_stay_inside_dataset_dir(self, anon_client, tmp_path):
        img = base64.b64encode(b"jpeg-bytes").decode()
        with (
            patch("smartwaste.web.DATASET_DIR", str(tmp_path)),
            patch("smartwaste.web.insert_entry", return_value=1),
        ):
            r = anon_client.post(
                "/api/report",
                json=self._payload(label="../../evil", bin_id="..\\..\\up", image_b64=img),
                headers=EDGE_HEADERS,
            )
        assert r.status_code == 200
        files = list(tmp_path.iterdir())
        # The image landed inside the dataset dir (a traversal would have
        # written outside tmp_path, leaving it empty).
        assert len(files) == 1
        assert files[0].parent == tmp_path

    def test_confidence_passed_through_to_db(self, anon_client):
        with patch("smartwaste.web.insert_entry", return_value=1) as mock_insert:
            anon_client.post(
                "/api/report",
                json=self._payload(confidence=0.66, llm_backend="gemini"),
                headers=EDGE_HEADERS,
            )
        entry = mock_insert.call_args[0][0]
        assert entry["confidence"] == pytest.approx(0.66)
        assert entry["llm_backend"] == "gemini"
