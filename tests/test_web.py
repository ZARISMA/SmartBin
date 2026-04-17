"""Tests for smartwaste/web.py — FastAPI endpoints."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from smartwaste.config import ADMIN_PASSWORD, ADMIN_USERNAME


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


class TestIndex:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_contains_title(self, client):
        r = client.get("/")
        assert "Smart Waste AI" in r.text


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
