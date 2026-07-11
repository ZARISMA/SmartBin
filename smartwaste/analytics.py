"""
smartwaste/analytics.py — period windows and payload assembly for /api/analytics.

Pure aggregation on top of the database range queries; web.py returns the
payload unchanged. Every ratio and delta is nullable so an empty database or
an empty previous window renders as "—" client-side instead of erroring.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from . import database as db
from .config import RECYCLABLE_CLASSES

# Stacked-series categories: "Empty" is excluded, unknown labels fold into Other.
CHART_CATEGORIES = ["Plastic", "Paper", "Glass", "Organic", "Aluminum", "Other"]

PERIODS = ("24h", "7d", "30d", "90d", "ytd")
PERIOD_LABELS = {
    "24h": "24 hours",
    "7d": "7 days",
    "30d": "30 days",
    "90d": "90 days",
    "ytd": "year to date",
}
_PERIOD_DAYS = {"7d": 7, "30d": 30, "90d": 90}

_FMT = "%Y-%m-%d %H:%M:%S"

EXPORT_COLUMNS = [
    "id",
    "timestamp",
    "bin_id",
    "label",
    "description",
    "brand_product",
    "location",
    "confidence",
    "llm_backend",
]
EXPORT_ROW_LIMIT = 10_000  # keeps the CSV response memory-bounded


@dataclass
class PeriodRange:
    start: datetime
    end: datetime
    prev_start: datetime
    prev_end: datetime
    granularity: str  # "hour" | "day"
    label: str

    def db_args(self) -> tuple[str, str]:
        return self.start.strftime(_FMT), self.end.strftime(_FMT)

    def prev_db_args(self) -> tuple[str, str]:
        return self.prev_start.strftime(_FMT), self.prev_end.strftime(_FMT)


def period_range(period: str, now: datetime | None = None) -> PeriodRange:
    """Compute the half-open [start, end) window for a period plus the
    equal-length window immediately before it.

    24h → hourly buckets ending in the current (partial) hour;
    7d/30d/90d → N daily buckets ending today (partial);
    ytd → daily buckets since Jan 1.
    """
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}")
    now = now or datetime.now()
    if period == "24h":
        start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=23)
        granularity = "hour"
    elif period == "ytd":
        start = datetime(now.year, 1, 1)
        granularity = "day"
    else:
        days = _PERIOD_DAYS[period]
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        granularity = "day"
    span = now - start
    return PeriodRange(
        start=start,
        end=now,
        prev_start=start - span,
        prev_end=start,
        granularity=granularity,
        label=PERIOD_LABELS[period],
    )


def make_buckets(start: datetime, end: datetime, granularity: str) -> list[dict]:
    """Generate the full [start, end) bucket axis with real date labels.

    Buckets come from Python, not SQL, so sparse or empty data still yields a
    complete axis. Keys match the DB bucket keys ('YYYY-MM-DD' / 'YYYY-MM-DD HH').
    """
    step = timedelta(hours=1) if granularity == "hour" else timedelta(days=1)
    key_fmt = "%Y-%m-%d %H" if granularity == "hour" else "%Y-%m-%d"
    label_fmt = "%H:00" if granularity == "hour" else "%b %d"
    out = []
    cur = start
    while cur < end:
        out.append({"key": cur.strftime(key_fmt), "label": cur.strftime(label_fmt)})
        cur += step
    if not out:  # zero-length window (e.g. YTD at exactly Jan 1 00:00:00)
        out.append({"key": start.strftime(key_fmt), "label": start.strftime(label_fmt)})
    return out


def _pct_delta(cur: int, prev: int) -> float | None:
    if not prev:
        return None
    return round((cur - prev) / prev * 100, 1)


def _diversion(counts: dict[str, int]) -> float | None:
    """Share of non-Empty classifications that landed in a recyclable category."""
    non_empty = sum(v for k, v in counts.items() if k != "Empty")
    if not non_empty:
        return None
    recyclable = sum(counts.get(c, 0) for c in RECYCLABLE_CLASSES)
    return round(recyclable / non_empty, 3)


def build_payload(period: str, now: datetime | None = None) -> dict:
    """Assemble the full /api/analytics response for one period."""
    rng = period_range(period, now)
    since, until = rng.db_args()
    prev_since, prev_until = rng.prev_db_args()

    cur = db.get_summary_in_range(since, until)
    prev = db.get_summary_in_range(prev_since, prev_until)
    by_cat = db.get_label_counts_in_range(since, until)
    by_cat_prev = db.get_label_counts_in_range(prev_since, prev_until)
    bin_counts = db.get_bin_counts_in_range(since, until)
    prev_bin_counts = db.get_bin_counts_in_range(prev_since, prev_until)

    diversion = _diversion(by_cat)
    prev_diversion = _diversion(by_cat_prev)
    avg_conf = cur["avg_confidence"]
    prev_conf = prev["avg_confidence"]

    buckets = make_buckets(rng.start, rng.end, rng.granularity)
    index = {b["key"]: i for i, b in enumerate(buckets)}
    data = {c: [0] * len(buckets) for c in CHART_CATEGORIES}
    for row in db.get_timeseries_in_range(since, until, rng.granularity):
        i = index.get(row["bucket"])
        if i is None or row["label"] == "Empty":
            continue
        cat = row["label"] if row["label"] in data else "Other"
        data[cat][i] += row["count"]

    kpis = {
        "total": {
            "value": cur["total"],
            "prev": prev["total"],
            "delta": _pct_delta(cur["total"], prev["total"]),
        },
        "diversion_rate": {
            "value": diversion,
            "prev": prev_diversion,
            # percentage-point difference, not relative change
            "delta": (
                round((diversion - prev_diversion) * 100, 1)
                if diversion is not None and prev_diversion is not None
                else None
            ),
        },
        "avg_confidence": {
            "value": round(avg_conf, 2) if avg_conf is not None else None,
            "prev": round(prev_conf, 2) if prev_conf is not None else None,
            "delta": (
                round(avg_conf - prev_conf, 2)
                if avg_conf is not None and prev_conf is not None
                else None
            ),
        },
        "active_bins": {
            "value": len(bin_counts),
            "prev": len(prev_bin_counts),
            "delta": len(bin_counts) - len(prev_bin_counts),
        },
    }

    leaderboard = [
        {"bin_id": b, "count": n}
        for b, n in sorted(bin_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "period": period,
        "period_label": rng.label,
        "granularity": rng.granularity,
        "range": {
            "start": since,
            "end": until,
            "prev_start": prev_since,
            "prev_end": prev_until,
        },
        "kpis": kpis,
        "series": {"buckets": buckets, "categories": CHART_CATEGORIES, "data": data},
        "by_category": by_cat,
        "by_category_prev": by_cat_prev,
        "leaderboard": leaderboard,
        "backends": db.get_backend_stats_in_range(since, until),
    }


def build_export_rows(period: str, now: datetime | None = None) -> list[dict]:
    """Rows for the period-scoped CSV export, newest first."""
    rng = period_range(period, now)
    since, until = rng.db_args()
    entries = db.get_entries(limit=EXPORT_ROW_LIMIT, since=since, until=until)
    return [{c: e.get(c) for c in EXPORT_COLUMNS} for e in entries]
