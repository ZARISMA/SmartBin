"""Tests for hexabin/analytics.py — period windows and payload assembly."""

from datetime import datetime

import pytest

from hexabin import analytics

NOW = datetime(2026, 7, 11, 14, 30, 7)


class TestPeriodRange:
    def test_24h_window(self):
        rng = analytics.period_range("24h", now=NOW)
        assert rng.granularity == "hour"
        assert rng.db_args() == ("2026-07-10 15:00:00", "2026-07-11 14:30:07")

    def test_7d_window(self):
        rng = analytics.period_range("7d", now=NOW)
        assert rng.granularity == "day"
        assert rng.db_args() == ("2026-07-05 00:00:00", "2026-07-11 14:30:07")

    def test_30d_window(self):
        assert analytics.period_range("30d", now=NOW).db_args()[0] == "2026-06-12 00:00:00"

    def test_90d_window(self):
        assert analytics.period_range("90d", now=NOW).db_args()[0] == "2026-04-13 00:00:00"

    def test_ytd_window(self):
        rng = analytics.period_range("ytd", now=NOW)
        assert rng.granularity == "day"
        assert rng.db_args()[0] == "2026-01-01 00:00:00"

    def test_prev_window_is_adjacent_and_equal_length(self):
        for period in analytics.PERIODS:
            rng = analytics.period_range(period, now=NOW)
            assert rng.prev_end == rng.start
            assert rng.end - rng.start == rng.prev_end - rng.prev_start

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            analytics.period_range("14d", now=NOW)


class TestMakeBuckets:
    def test_24_hourly_buckets(self):
        rng = analytics.period_range("24h", now=NOW)
        buckets = analytics.make_buckets(rng.start, rng.end, "hour")
        assert len(buckets) == 24
        assert buckets[0] == {"key": "2026-07-10 15", "label": "15:00"}
        assert buckets[-1]["key"] == "2026-07-11 14"

    def test_7_daily_buckets_with_date_labels(self):
        rng = analytics.period_range("7d", now=NOW)
        buckets = analytics.make_buckets(rng.start, rng.end, "day")
        assert len(buckets) == 7
        assert buckets[0] == {"key": "2026-07-05", "label": "Jul 05"}
        assert buckets[-1]["key"] == "2026-07-11"

    def test_zero_length_window_yields_one_bucket(self):
        jan1 = datetime(2026, 1, 1)
        assert len(analytics.make_buckets(jan1, jan1, "day")) == 1


class TestBuildPayload:
    def _patch_db(self, monkeypatch, **overrides):
        defaults = {
            "get_summary_in_range": lambda since, until: {"total": 0, "avg_confidence": None},
            "get_label_counts_in_range": lambda since, until: {},
            "get_bin_counts_in_range": lambda since, until: {},
            "get_timeseries_in_range": lambda since, until, granularity="day": [],
            "get_backend_stats_in_range": lambda since, until: [],
        }
        defaults.update(overrides)
        for name, fn in defaults.items():
            monkeypatch.setattr(analytics.db, name, fn)

    def test_kpi_math_against_previous_window(self, monkeypatch):
        cur_since = analytics.period_range("7d", now=NOW).db_args()[0]

        def summary(since, until):
            if since == cur_since:
                return {"total": 30, "avg_confidence": 0.9}
            return {"total": 20, "avg_confidence": 0.8}

        def labels(since, until):
            if since == cur_since:
                return {"Plastic": 20, "Other": 5, "Empty": 5}  # diversion 20/25 = 0.8
            return {"Plastic": 10, "Other": 10}  # prev diversion 0.5

        def bins(since, until):
            return {"bin-01": 25, "bin-02": 5} if since == cur_since else {"bin-01": 20}

        self._patch_db(
            monkeypatch,
            get_summary_in_range=summary,
            get_label_counts_in_range=labels,
            get_bin_counts_in_range=bins,
        )
        p = analytics.build_payload("7d", now=NOW)
        assert p["kpis"]["total"] == {"value": 30, "prev": 20, "delta": 50.0}
        assert p["kpis"]["diversion_rate"]["value"] == pytest.approx(0.8)
        assert p["kpis"]["diversion_rate"]["delta"] == pytest.approx(30.0)  # points
        assert p["kpis"]["avg_confidence"]["delta"] == pytest.approx(0.1)
        assert p["kpis"]["active_bins"] == {"value": 2, "prev": 1, "delta": 1}
        assert p["leaderboard"] == [
            {"bin_id": "bin-01", "count": 25},
            {"bin_id": "bin-02", "count": 5},
        ]

    def test_empty_db_yields_nulls_not_errors(self, monkeypatch):
        self._patch_db(monkeypatch)
        p = analytics.build_payload("7d", now=NOW)
        k = p["kpis"]
        assert k["total"] == {"value": 0, "prev": 0, "delta": None}
        assert k["diversion_rate"] == {"value": None, "prev": None, "delta": None}
        assert k["avg_confidence"] == {"value": None, "prev": None, "delta": None}
        assert k["active_bins"] == {"value": 0, "prev": 0, "delta": 0}
        assert len(p["series"]["buckets"]) == 7
        assert all(v == [0] * 7 for v in p["series"]["data"].values())
        assert p["leaderboard"] == []

    def test_series_folds_unknown_drops_empty_and_out_of_axis(self, monkeypatch):
        def ts(since, until, granularity="day"):
            return [
                {"bucket": "2026-07-05", "label": "Plastic", "count": 3},
                {"bucket": "2026-07-05", "label": "Mystery", "count": 2},
                {"bucket": "2026-07-05", "label": "Empty", "count": 9},
                {"bucket": "1999-01-01", "label": "Plastic", "count": 7},
            ]

        self._patch_db(monkeypatch, get_timeseries_in_range=ts)
        p = analytics.build_payload("7d", now=NOW)
        data = p["series"]["data"]
        assert data["Plastic"] == [3, 0, 0, 0, 0, 0, 0]
        assert data["Other"] == [2, 0, 0, 0, 0, 0, 0]
        assert "Empty" not in data

    def test_by_category_keeps_empty_for_the_donut_payload(self, monkeypatch):
        self._patch_db(
            monkeypatch,
            get_label_counts_in_range=lambda since, until: {"Plastic": 1, "Empty": 4},
        )
        p = analytics.build_payload("7d", now=NOW)
        assert p["by_category"] == {"Plastic": 1, "Empty": 4}


class TestBuildExportRows:
    def test_projects_export_columns_only(self, monkeypatch):
        entry = {c: "x" for c in analytics.EXPORT_COLUMNS}
        entry["filename"] = "should-not-appear"
        monkeypatch.setattr(analytics.db, "get_entries", lambda **kw: [entry])
        rows = analytics.build_export_rows("7d", now=NOW)
        assert list(rows[0].keys()) == analytics.EXPORT_COLUMNS

    def test_passes_period_range_to_db(self, monkeypatch):
        captured = {}

        def fake_get_entries(**kw):
            captured.update(kw)
            return []

        monkeypatch.setattr(analytics.db, "get_entries", fake_get_entries)
        analytics.build_export_rows("7d", now=NOW)
        assert captured["since"] == "2026-07-05 00:00:00"
        assert captured["until"] == "2026-07-11 14:30:07"
        assert captured["limit"] == analytics.EXPORT_ROW_LIMIT
