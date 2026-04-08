# SmartBin Code Review — Production Readiness Assessment

**Date:** 2026-04-06
**Scope:** Full codebase review — dual-camera waste classification system
**Supersedes:** 2026-03-20 review

---

## Overall Verdict

The project has matured significantly since the March 2026 review. The core architecture is solid: clean module separation, Pydantic config layering, sensor fusion, a full test suite, CI/CD, and Docker. The gap between "working prototype" and "product a corporation would ship" has narrowed — but it remains. This review covers where the project stands today and what it takes to close that gap.

---

## Progress Since Last Review

The following items from the old P0/P1 list have been addressed:

| Item | Status |
|------|--------|
| No tests | **Done** — 2,210 lines across 14 test modules, thread-safety tests, edge cases |
| No CI/CD | **Done** — GitHub Actions: ruff, mypy, pytest --cov |
| No config system | **Done** — Pydantic BaseSettings, 50+ env-var-overridable constants |
| No `pyproject.toml` | **Done** — CLI entry points, ruff/mypy/pytest config |
| No retry/backoff | **Done** — tenacity + circuit breaker with configurable thresholds |
| No Docker | **Done** — docker-compose with app + PostgreSQL + Grafana |
| No web UI | **Done** — FastAPI + MJPEG stream + stats + history |
| Duplicated entry points | **Done** — strategies pattern (ManualStrategy, PresenceGateStrategy) |
| OAK-D depth unused | **Done** — `oak_native.py`: depth + IMU + MobileNetSSD voting |
| Fake sensor data unlabeled | **Done** — columns are now `simulated_*` in all backends |
| Excel/JSON write paths | **Done** — SQLite is sole source of truth |
| Location hardcoded | **Done** — `LOCATION` configurable via env var |

---

## 1. CRITICAL — Fix Before Any Deployment

### 1.1 Thread Safety: `toggle_auto()` Still Unprotected

**`state.py` — `toggle_auto()` reads and writes `self.auto_classify` without holding `self._lock`.** The classify daemon threads read `auto_classify` via `get_display()` which _does_ acquire the lock. This is a live data race that can cause inconsistent auto-mode behavior under load.

**Fix:** Wrap the read-modify-write in `toggle_auto()` with `with self._lock:`. Three lines.

### 1.2 No Authentication on the Web UI or API

Every FastAPI endpoint — including `/api/classify` and `/api/toggle-auto` — is open to anyone on the network. A guest on the same Wi-Fi can trigger classifications, disable auto mode, or scrape the full entry history.

**Action:**
- Add an `X-API-Key` header check as FastAPI middleware (simplest approach)
- Set the key via `API_KEY` env var; 401 if missing or wrong
- Alternatively, issue a short-lived JWT on a `/login` endpoint and validate the bearer token

### 1.3 Confidence Scores Still Not Implemented

This was flagged in the last review and remains unaddressed. Gemini's output is accepted as ground truth regardless of how uncertain the model was. An ambiguous item (e.g., mixed paper/plastic packaging) gets the same confidence treatment as an obvious aluminum can.

**Action:** Add `"confidence": 0.0-1.0` to the prompt's expected JSON schema. In `classifier.py`, treat results below a threshold (e.g., 0.65) as `"Other"` and log a warning. Store confidence in the database.

---

## 2. HIGH — Architecture & Engineering

### 2.1 Polling Architecture Does Not Scale

The frontend polls `/api/state` every 1 second, `/api/stats` every 5 seconds. With 10 concurrent browser clients, that's 10 requests/second of overhead delivering nothing new most of the time.

**Action:** Add a FastAPI `WebSocket` endpoint for classification events. The camera loop broadcasts a JSON message after each classification; clients subscribe once. MJPEG stream stays as-is (that's already push-based).

### 2.2 No Local/Offline Fallback Model

The system has 100% dependency on Google Gemini. A connectivity blip in an Armenian municipal building means zero classifications until the connection recovers.

**Roadmap (already described in old review — still not started):**
1. Collect labeled images from `waste_dataset/` (already accumulating)
2. Train MobileNetV2 classifier; convert to OpenVINO blob for OAK VPU
3. Run on-device for core categories (Plastic/Glass/Paper/Organic/Aluminum)
4. Use Gemini only for low-confidence predictions and new brand/variant recognition

This is the single highest-leverage engineering investment remaining. It eliminates API costs for routine classifications, enables offline operation, and creates a proprietary training flywheel.

### 2.3 No Health Check for the App Container

`docker-compose.yml` defines a `pg_isready` health check for PostgreSQL but nothing for the FastAPI app. A container that started but crashed silently after 30 seconds shows as healthy.

**Action:** Add `GET /health` (liveness — is the process alive?) and `GET /ready` (readiness — are cameras connected and DB reachable?) to `web.py`. Add `healthcheck` in docker-compose pointing at `/health`.

### 2.4 No Rate Limiting or CORS

Nothing prevents a client from calling `/api/classify` in a tight loop, exhausting the Gemini quota in minutes. There are also no CORS headers, so cross-origin requests from any domain succeed.

**Action:**
- Add `slowapi` rate limiting (e.g., 10 req/min on `/api/classify`)
- Add `CORSMiddleware` with an explicit origin whitelist via `CORS_ORIGINS` env var

### 2.5 No HTTPS

The web UI and API run over plain HTTP. Any API key or session token transmitted is cleartext on the network.

**Action:** Add an `nginx` or `caddy` service to docker-compose as TLS-terminating reverse proxy. Self-signed cert for LAN deployment; Let's Encrypt for any public-facing deployment.

### 2.6 Temporal Reasoning Absent

Each frame is classified independently. A blurry or transitional frame (item mid-drop) gets the same weight as a sharp, stable frame.

**Action:** Buffer the last 3 classification results. If they don't agree within the same top-level category, emit the majority vote (or abstain and log `"Uncertain"`).

---

## 3. MEDIUM — Data, Monitoring & Observability

### 3.1 No Prometheus Metrics

The Grafana dashboard queries PostgreSQL for business data but exposes zero operational metrics: no API call latency, no circuit breaker state, no classification throughput, no error rate.

**Action:** Add `prometheus-client` and expose `/metrics`. Key metrics to instrument:
- `smartwaste_classifications_total` (counter, labels: category, result)
- `smartwaste_api_latency_seconds` (histogram)
- `smartwaste_circuit_breaker_open` (gauge)
- `smartwaste_frames_captured_total` (counter, labels: camera)

Add a Prometheus service to docker-compose; update Grafana to use it alongside PostgreSQL.

### 3.2 No Database Migration Framework

Schema DDL is embedded in `database.py`. Adding a column means editing code and running a manual `ALTER TABLE`. There is no migration history, no rollback, and no way to verify which schema version is deployed.

**Action:** Adopt Alembic. Current schema becomes revision `0001_initial`. Future changes are versioned migrations. The app runs `alembic upgrade head` at startup (or in a separate init container).

### 3.3 No Data Retention Policy

Images and database rows accumulate indefinitely. A Raspberry Pi SD card with months of 24/7 operation will fill up without warning.

**Action:**
- Add `DATA_RETENTION_DAYS` setting (default: 90)
- Add a cleanup function in `dataset.py`: delete images and DB rows older than retention window
- Call it as a background task (FastAPI `BackgroundTasks` or a simple APScheduler job) daily

### 3.4 No OpenAPI Documentation

FastAPI generates `/docs` and `/redoc` automatically — but only if endpoints have Pydantic response models. Currently, all endpoints return raw dicts with no schema declaration.

**Action:** Define Pydantic response models (`StateResponse`, `EntryResponse`, `StatsResponse`) and add them to all `@app.get`/`@app.post` decorators. Costs ~30 lines; gives a free interactive API explorer.

### 3.5 No Structured Logging

Logs go to rotating file + stdout as plain text. Feeding them to Loki, ELK, or Datadog requires fragile regex parsing.

**Action:** Add `python-json-logger` and configure `log_setup.py` to emit JSON by default (enable via `LOG_FORMAT=json` env var). Keep human-readable format for local dev (default).

### 3.6 Weight Field Is Always Empty

Every database row has `weight: ""`. This looks like a data bug to anyone querying the database. Either hook up an HX711 load cell (sub-$5 component with Pi GPIO) or remove the column entirely.

### 3.7 Prompt Has No Version Tracking

When the prompt in `prompt.py` changes, there's no way to attribute a shift in classification accuracy to the prompt change versus data drift.

**Action:** Add `PROMPT_VERSION = "v1.2"` to `config.py`. Log it with every classification. Add a `prompt_version` column to the database. Then Grafana can segment accuracy by prompt version.

---

## 4. Competitive Gap Analysis — What Big Company Projects Have

This section maps the gap between SmartBin's current state and what commercially-deployed waste AI products ship (AMP Robotics, ZenRobotics, Bin-e, Greyparrot).

### 4.1 Multi-Bin Fleet Management

**Enterprise products:** Central dashboard managing hundreds of bins. Per-bin status, alerts, historical data, map view.

**SmartBin now:** Single device, local UI only. No concept of a "fleet."

**Gap:** Add a lightweight cloud relay service (FastAPI + PostgreSQL). Each bin authenticates and POSTs classifications. Central Grafana shows per-bin dashboards. Use MQTT as the protocol — it's designed for IoT fleet telemetry and handles intermittent connectivity natively.

### 4.2 Offline-First Edge Architecture

**Enterprise products:** Function fully offline. Cloud sync when available.

**SmartBin now:** Fails completely when Gemini is unreachable.

**Gap:** See section 2.2. On-device model is not optional for commercial deployment — it's a hard requirement in most real-world settings.

### 4.3 Gamification & User Behavior Change

**Enterprise products (Recycle Coach, Rubicon):** Points, streaks, environmental impact equivalents ("you've recycled the equivalent of X plastic bottles"). Drives behavior change which is the actual product goal.

**SmartBin now:** No user-facing feedback beyond the classification label.

**Gap:** Add a "session impact" display to the web UI. Show: items classified this session, estimated CO2 offset (use a lookup table by category), cumulative totals. No backend changes required — compute from existing data.

### 4.4 Regulatory Compliance & Audit Trail

**Enterprise products:** Tamper-evident audit logs, GDPR data deletion, data export for regulatory reporting, chain-of-custody for waste streams.

**SmartBin now:** No audit log, no deletion API, no export.

**Gap:**
- Add an `audit_log` table (action, actor, timestamp, entry_id) — append-only
- Add `DELETE /api/entries/{id}` for GDPR right-to-erasure
- Add `GET /api/export?format=csv&from=&to=` for regulatory export

### 4.5 Mobile App / PWA

**Enterprise products:** iOS/Android apps for bin operators. Push notifications when bins are full.

**SmartBin now:** Browser-only; no mobile optimization.

**Gap:** Add a PWA manifest (`manifest.json`) + service worker to `web_static/`. Costs 1-2 hours; enables "Add to Home Screen" on iOS/Android, offline shell, push notification infrastructure.

### 4.6 ERP/Building Management System Integration

**AMP Robotics, Greyparrot:** Integrate with SAP, Oracle, BMS platforms. Waste haulers receive automated pickup requests when fill level exceeds threshold.

**SmartBin now:** No integration surface.

**Gap:** Add a webhook system. Configure `WEBHOOK_URL` + `WEBHOOK_EVENTS` (e.g., `bin_full,circuit_open`). POST JSON payload on event. This is the minimum integration surface for a B2B sale.

### 4.7 Predictive Analytics

**Enterprise products:** Forecast when a bin will reach capacity. Optimize pickup schedules. Reduce unnecessary collections by 30-40%.

**SmartBin now:** Depth sensor data exists (in oak_native.py) but is only used for presence detection; no fill-level time series stored.

**Gap:** Store depth-derived fill percentage in the database on each classification. After 2-4 weeks of data, a simple linear regression per bin can forecast fill time. Expose as `GET /api/predictions/fill-time`.

### 4.8 Internationalization

**Enterprise products:** Multi-language UI, locale-specific regulatory mappings, regional brand recognition.

**SmartBin now:** English-only UI. Armenian brand examples in prompt (good product thinking, but hardcoded).

**Gap:** Extract UI strings to a JSON locale file. Add a `LOCALE` env var. Move brand hint examples in `prompt.py` to locale-specific config files loaded at startup.

### 4.9 Regulatory Category Mapping

**EU, US, and many markets require waste classified to specific regulatory codes** (EU Waste Catalogue LoW codes, EPA categories, etc.) for compliance reporting.

**SmartBin now:** 7 internal categories. No mapping to any standard.

**Gap:** Add a `REGULATORY_MAPPING` config (JSON dict per region) that maps `Plastic → 15 01 02` (EU LoW). Include in export and audit log. Configurable per deployment without code change.

### 4.10 Hardware Failure Alerting

**Enterprise products:** Ops team gets paged when a camera goes offline, when a model stops responding, when disk fills.

**SmartBin now:** Circuit breaker opens silently (logged, not alerted). No disk monitoring. No camera watchdog.

**Gap:** Add alerting hooks to the circuit breaker's open/close state transition, to camera reconnect failure, and to a disk-space check. Route alerts to a `ALERT_WEBHOOK_URL` (Slack, PagerDuty, email relay — caller's choice).

---

## 5. Updated Priority Action Plan

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Fix `toggle_auto()` thread safety (`state.py`) | XS | Data integrity |
| P0 | Add web UI/API authentication (API key middleware) | S | Security baseline |
| P0 | Add confidence score to Gemini prompt + threshold logic | S | Classification quality |
| P1 | Add `/health` + `/ready` endpoints + docker-compose healthcheck | XS | Container orchestration |
| P1 | Add rate limiting (SlowAPI) + CORS middleware | S | Security |
| P1 | Prometheus `/metrics` endpoint + docker-compose Prometheus service | M | Operational visibility |
| P1 | WebSocket push for classification events | M | Real-time UX, scalability |
| P2 | Data retention: `DATA_RETENTION_DAYS` setting + daily cleanup | S | Disk management |
| P2 | HTTPS: nginx/Caddy TLS termination in docker-compose | S | Security |
| P2 | OpenAPI response models (Pydantic) on all endpoints | S | API documentation |
| P2 | Structured JSON logging via `python-json-logger` | S | Log aggregation |
| P2 | Prompt version tracking (`PROMPT_VERSION` in config + DB) | XS | Experiment traceability |
| P2 | Weight field: integrate HX711 load cell or drop the column | XS | Data quality |
| P2 | Alembic migration framework | M | Schema evolution |
| P2 | Session impact display in web UI (CO2 offset, totals) | M | User engagement |
| P3 | Local TFLite/OpenVINO on-device model (offline fallback) | XL | Offline capability, cost -80% |
| P3 | Fleet management backend (MQTT/REST + central dashboard) | XL | Commercial scalability |
| P3 | Webhook system for events (bin_full, circuit_open, disk_low) | M | Operations, B2B integration |
| P3 | PWA manifest + service worker | M | Mobile UX |
| P3 | Predictive fill-time analytics (linear regression on depth data) | L | Enterprise differentiator |
| P3 | Data export: `GET /api/export?format=csv` | S | Compliance |
| P3 | GDPR deletion: `DELETE /api/entries/{id}` + audit log | S | Compliance |
| P3 | Regulatory category mapping (EU LoW codes, etc.) | M | Enterprise compliance |
| P3 | i18n: locale files + `LOCALE` env var | M | International deployment |

---

## 6. What the Project Is Doing Right

Don't lose these strengths while improving:

- **Clean module separation** — each file has a single responsibility. Strategies, presence detection, sensor fusion, classifier, and database are genuinely decoupled. This is better than most prototypes.
- **Presence-gated API calls** — `PresenceDetector` is a smart cost optimization that most teams wouldn't think of until they see the Gemini bill.
- **Dual-camera approach** — two angles reduce classification ambiguity. Genuine hardware differentiator.
- **Sensor fusion (`oak_native.py`)** — depth + IMU + on-device NN voting is rare in open-source IoT. This is publication-worthy engineering.
- **Pydantic config layering** — 50+ constants overridable via env vars without code changes. Enterprise-grade.
- **Armenian brand recognition** — domain-specific prompt tuning for the deployment location signals product thinking, not just engineering.
- **Progressive dataset accumulation** — every Gemini-labeled classification feeds a future on-device model. The flywheel is running.
- **CI/CD pipeline** — ruff + mypy + pytest --cov on every push. Most startups skip this until regressions become a crisis.
- **Docker Compose + Grafana** — operations infrastructure from day one.
- **Thread-safe AppState** — proper lock-based classify gating shows concurrency awareness. Fix the one gap noted above and it's solid.
