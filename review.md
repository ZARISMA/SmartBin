# SmartBin — Competitive Review & Enterprise Gap Analysis

*Reviewed 2026-04-17 against SmartBin v0.1.0 (branch `master`).*
*Benchmarks: Bigbelly (fleet hardware), Nordsense (wireless fill sensors), Compology (vision for dumpsters), Greyparrot AI (MRF-line classification), CleanRobotics TrashBot (autonomous sorting), Recycleye (robotic picking), Enevo (route optimization).*

---

## 1. Executive Summary

SmartBin is a **working, visually credible MVP** with a distinctive technical angle (dual OAK-D stereo cameras + Gemini vision + edge-to-server topology) and a marketing site polished enough to pass for Series-A. It is **not yet an enterprise product.** The gap to Bigbelly/Nordsense/Compology is not the classifier — it's everything *around* the classifier: multi-tenancy, security, observability, fleet management, analytics, compliance.

The three shortest honest sentences:

1. **A single leaked `.env` compromises every deployed bin.** Auth is one shared admin password; the edge API key is a single shared secret; `INSTRUCTIONS.md` commits both in plaintext; the admin password also doubles as a valid Bearer token (`smartwaste/web.py:96`).
2. **The product's headline metric — fill level — is faked.** `weight` is a never-populated string (`smartwaste/schemas.py:18`, `smartwaste/dataset.py:37`); all environmental sensors are `random.uniform()` (`smartwaste/dataset.py:15-22`). Procurement will catch this in the first demo.
3. **One vendor outage = full service outage.** Every classification path depends on Google Gemini (`smartwaste/classifier.py:27,58`). No on-device fallback, no multi-model strategy, no cost telemetry.

None of this is fatal. All of it is fixable in roughly **90 days** of focused work (Section 6). SmartBin's open-source stack, OAK-native sensor fusion, and clean codebase are real advantages — they just need to be wrapped in the operational scaffolding every enterprise buyer expects.

**Bottom line for investors:** viable pilot product today, 3 months from being credibly enterprise-ready, 6 months from being competitively differentiated. The engineering is ahead of the operations.

---

## 2. Competitive Landscape

| Player | What they do well | SmartBin's gap |
|---|---|---|
| **Bigbelly** | Solar-powered fleet of 75k+ connected bins; mature fleet mgmt, OTA, SLA contracts with cities. | No OTA, no fleet hierarchy, no SLA tracking. |
| **Nordsense** | Ultrasonic fill-level sensors + route-optimization API; municipal integrations. | No real fill sensor; no route API; no municipal ERP hooks. |
| **Compology** | Camera-in-dumpster + contamination scoring + weekly PDF reports to haulers. | No contamination score, no report export, no hauler portal. |
| **Greyparrot AI** | MRF conveyor vision, ISO 27001, multi-language dashboards, on-prem option. | No certifications, no i18n, Gemini-cloud only. |
| **CleanRobotics TrashBot** | Autonomous lid + motorized sorting. | Vision-only, no actuation. |
| **Enevo** | Collection route optimization from historical fill data. | No time-series analytics, no forecasting. |

SmartBin's **edge**: open-source, commodity hardware (OAK-D + Raspberry Pi), on-device sensor fusion (`smartwaste/oak_native.py`), fast iteration. Preserve these while closing the operational gap.

---

## 3. Severity Legend

- **BLOCKER** — prevents sale to any enterprise or municipal buyer. Must fix before go-to-market.
- **MAJOR** — competitor parity requirement. Missing it means losing deals you'd otherwise win.
- **MINOR** — polish / trust signal. Fix before a Series A or a keynote demo.

---

## 4. Gap Analysis

### 4.1 Backend Architecture & API

**What exists today.** FastAPI service (`smartwaste/web.py`) with session-cookie auth (`:81`), Bearer-token auth for edges (`:89–98`), an in-memory bin registry (`_bin_registry: dict` at `:113`), thirteen endpoints covering login/dashboard/stream/classify/state/entries/stats/bins/report/heartbeat. Pydantic-typed edge payloads (`smartwaste/schemas.py`). Async lifespan starts the camera thread and (if `EDGE_MODE`) a heartbeat thread.

**Gaps vs. enterprise.**
- **[BLOCKER] No multi-tenancy.** One admin account (`smartwaste/settings.py:105-106`, default `admin`/`password123`). No organizations, districts, users, roles, or per-tenant scoping. A single customer = the whole deployment.
- **[BLOCKER] Admin password is a valid Bearer token.** `smartwaste/web.py:96` — `token == ADMIN_PASSWORD or token == EDGE_API_KEY`. Whoever logs in can also hit the edge API; whoever sniffs an edge request can log in.
- **[BLOCKER] Bin registry is volatile.** `_bin_registry` at `smartwaste/web.py:113` is a plain `dict` — `docker compose restart app` wipes the known-fleet topology until every bin re-heartbeats.
- **[MAJOR] No API versioning.** `/api/report` has no `v1/` prefix; one breaking change forces a fleet-wide firmware push.
- **[MAJOR] No rate limiting.** `/api/report` and `/api/heartbeat` accept unlimited POSTs with no throttling. Any leaked edge key can DoS the server.
- **[MAJOR] No audit log / request log.** No record of who classified, logged in, exported, or mutated state.
- **[MINOR] No OpenAPI customization / client SDKs.** FastAPI auto-docs exist but no published, versioned SDK for integrators.

**Remediation.** Add a `tenants`/`users`/`api_keys` schema; replace the dual-use auth with per-user sessions (Argon2 hashes) and per-device API keys issued from an admin UI. Persist the bin registry to Postgres. Add `slowapi` for rate limits, `fastapi_versioning` for `/api/v1/*`, and a middleware that writes audit events (`actor_id, action, resource, ip, ts`) to an append-only table.

**Effort.** 2–3 weeks for one backend engineer.

---

### 4.2 ML / AI Pipeline

**What exists today.** Cloud-only Gemini classification (`smartwaste/classifier.py:27` imports `google.genai`; `:58` builds the client at module import, failing hard if no key). Tenacity exponential-backoff retries (`:132-142`) and a thread-safe circuit breaker (`:67-100`). Strict JSON schema with 7 fixed categories (`smartwaste/prompt.py`). Separate OAK-native sensor fusion pipeline (`smartwaste/oak_native.py`) combining RGB + depth ROI + IMU shock detection + MobileNet-SSD voting — but the final classification label still comes from Gemini.

**Gaps vs. enterprise.**
- **[BLOCKER] Single-vendor lock-in.** Gemini goes down → every SmartBin worldwide stops classifying. No local fallback, no multi-model fan-out, no "safe default" category.
- **[BLOCKER] Privacy / data-residency.** Every image leaves the LAN and hits Google servers. No on-device option = immediate disqualification for EU municipal contracts, hospitals, defense sites.
- **[MAJOR] No confidence / contamination score.** The prompt (`smartwaste/prompt.py:17-23`) returns `category` / `description` / `brand_product` but no probability or "mixed-waste ratio." Compology's flagship metric is *contamination %*; SmartBin cannot produce it.
- **[MAJOR] Weight & volume are unused.** `EdgeReport.weight: str = ""` (`smartwaste/schemas.py:18`), and `save_entry()` hard-codes `"weight": ""` (`smartwaste/dataset.py:37`). The schema looks production-ready; the data is blank.
- **[MAJOR] Armenian brands hard-coded in the prompt.** `smartwaste/prompt.py:22` mentions "Jermuk, Bjni, BOOM" — untouchable without a code change. Doesn't generalize to any other market.
- **[MAJOR] No dataset / feedback loop.** Images save to `waste_dataset/*.jpg` but there is no labeling UI, no active-learning queue, no way for an operator to correct a misclassification and improve the model.
- **[MAJOR] No model pinning / cost telemetry.** `MODEL_NAME = "gemini-3-flash-preview"` is a moving target; no per-call token or USD cost is logged.
- **[MINOR] Classifier client built at import time** (`:58`). Unit tests must stub Gemini or module import crashes without `GEMINI_API_KEY`.

**Remediation.** Ship a small on-device fallback (`ultralytics YOLOv8-n` or MobileNet-SSD already loaded in OAK) that runs when circuit is open; return a `confidence: float` and `contamination_pct: float` field; pin `model_name` per deployment; extract brand list to `config/brands/<region>.yaml`; add a `corrections` table + minimal label-fix UI; log every Gemini call with input/output token counts and computed USD cost.

**Effort.** 4–6 weeks (includes retraining the on-device fallback on the saved `waste_dataset/`).

---

### 4.3 Hardware & Device Fleet Management

**What exists today.** Entry points for dual OAK (`main.py`), auto-gate presence (`mainauto.py`), OAK-native sensor fusion (`mainoak.py`), Raspberry Pi dual-camera (`mainraspberry.py`). Heartbeat every 30 s (`settings.py:116`); bin flips offline after 60 s silence (`smartwaste/web.py:116`). Docker Compose variants for server, full-edge, and lightweight-edge.

**Gaps vs. enterprise.**
- **[BLOCKER] No OTA updates.** Firmware/code lives in the Docker image. Pushing a fix to 100 deployed bins = 100 SSH sessions. Bigbelly ships firmware updates from a cloud console.
- **[BLOCKER] Shared edge API key.** Every device uses the same `SMARTWASTE_EDGE_API_KEY`. One key leak = whole fleet compromise, no revocation path. No per-device certificate, no mTLS.
- **[BLOCKER] No real fill-level sensor.** Environmental fields are `random.uniform()` (`smartwaste/dataset.py:15-22`). "Volume sensor" does not exist in the repo. Ultrasonic ToF + weight cell is the Nordsense commodity offering.
- **[MAJOR] No fleet topology.** Bins are a flat list — no zones, routes, districts, depots, or device groups.
- **[MAJOR] Health is one signal.** Only "heartbeat seen in last 60 s." No CPU/RAM/disk, USB-camera-up, Gemini-key-valid, DB-reachable, uptime distribution.
- **[MAJOR] No remote reboot / diagnostics.** Field engineer must physically visit.
- **[MAJOR] No factory provisioning flow.** A new Pi has to be hand-edited (`INSTRUCTIONS.md:122-128`) — no QR-scan-to-enroll, no zero-touch.
- **[MINOR] No lid actuation / access control.** CleanRobotics TrashBot can refuse to open for wrong waste; SmartBin cannot.

**Remediation.** Adopt **balena**, **Mender**, or **AWS IoT Greengrass** for OTA + diagnostics. Issue per-device X.509 certs via a tiny CA; rotate via API. Add a real `VL53L1X` ToF sensor + HX711 load cell over I²C on the Pi; expose `weight_g` / `fill_pct` in `EdgeReport`. Introduce a `devices`/`device_groups` table; add `/api/v1/devices/:id/reboot`, `/api/v1/devices/:id/diagnostics` endpoints.

**Effort.** 4–8 weeks (hardware BOM + CI + OTA infra).

---

### 4.4 Data, Analytics & Reporting

**What exists today.** Single flat table `waste_entries` in SQLite or Postgres (`smartwaste/database.py:27-65`). Columns cover label/description/brand/timestamp/weight/bin_id + five `simulated_*` env fields. Indexes on label and timestamp (Postgres only, `:63-64`). Threaded Postgres pool with `maxconn=5` (`:109-117`). Grafana mounted as a sidecar.

**Gaps vs. enterprise.**
- **[BLOCKER] No CSV / PDF / webhook export.** City council reporting, hauler invoicing, ESG audits all require export. Compology sends weekly PDFs; SmartBin has no `/api/export` of any kind.
- **[BLOCKER] `bin_id` is not a tenant key.** `bin_id TEXT DEFAULT 'bin-01'` (`:42`) — nullable, no FK to an `organizations` table, no row-level security. Every logged-in user sees every bin's data.
- **[BLOCKER] Headline metric is simulated.** Analytics built on `simulated_temperature`, `simulated_humidity`, `simulated_vibration`, `simulated_air_pollution`, `simulated_smoke` (`smartwaste/dataset.py:15-22`) are demo-only. Any data scientist who opens Grafana will spot the `random.uniform()` distribution inside a week.
- **[MAJOR] No time-series schema.** Fill-rate curves, collection-interval forecasts, anomaly detection all require time-bucketed rollups. The repo has none.
- **[MAJOR] No retention policy.** `waste_dataset/*.jpg` grows unbounded; Postgres volume grows unbounded.
- **[MAJOR] No schema migrations.** `CREATE TABLE IF NOT EXISTS` only (`:27, :47`). First schema change in production = manual DDL. No Alembic, no Flyway.
- **[MAJOR] Postgres pool is tiny.** `maxconn=5` (`:111`) will deadlock under any concurrent load.
- **[MAJOR] No GDPR subject-access / delete.** No endpoint to export or purge an individual's or a tenant's data.

**Remediation.** Introduce Alembic migrations; add `organizations`, `users`, `device_groups`, `daily_bin_stats` (time-bucketed) tables with `org_id` FKs; wire Postgres RLS. Add `/api/v1/export?format=csv|pdf&range=7d` using `pandas` + `weasyprint`. Swap simulated env fields for a `sensors` JSONB column populated only by real sensors, or delete them. Add a nightly image-retention job (`delete where created_at < now() - :days`). Raise pool `maxconn` to ≥ (2 × gunicorn workers).

**Effort.** 3–4 weeks.

---

### 4.5 Security & Compliance

**What exists today.** Session middleware with configurable secret (`smartwaste/settings.py:107`). Bearer auth for edges. `.env`-based secrets.

**Gaps vs. enterprise. This section is the single biggest risk.**
- **[BLOCKER] Secrets committed to the repo.** `INSTRUCTIONS.md:6` — Pi SSH password `Hexa1234`. `INSTRUCTIONS.md:8` — edge API key `smartbin-edge-2026-a7f3k9` *and* admin credentials `admin`/`password123`. `INSTRUCTIONS.md:80` repeats the key in a shell snippet. These are now in git history, *forever*, and visible to anyone with repo read access.
- **[BLOCKER] Default session secret shipped in source.** `secret_key: str = "smartwaste-session-secret-change-in-prod"` (`smartwaste/settings.py:107`). Session forgery is trivial unless the operator happens to set `SMARTWASTE_SECRET_KEY`.
- **[BLOCKER] Default admin credentials.** `admin`/`password123` (`smartwaste/settings.py:105-106`). Ships enabled.
- **[BLOCKER] Admin password doubles as Bearer token.** `smartwaste/web.py:96`. Either credential compromises both surfaces.
- **[BLOCKER] No TLS.** FastAPI served on plain HTTP `0.0.0.0:8000` (`smartwaste/settings.py:100-101`; `INSTRUCTIONS.md:5` uses `http://`). Credentials, session cookies, and edge keys cross the LAN in clear text.
- **[MAJOR] Grafana ships as `admin/admin`.** `docker-compose.yml:46` defaults `GF_SECURITY_ADMIN_PASSWORD` to `admin`.
- **[MAJOR] DB password `smartwaste`** default (`smartwaste/settings.py:97`, `docker-compose.yml:13`), connection unencrypted (`sslmode=disable`).
- **[MAJOR] No secrets management.** `.env` files only — no Vault, no AWS Secrets Manager, no SOPS, no K8s Secret CSI.
- **[MAJOR] No dependency scanning.** No Dependabot/Renovate config, no `pip-audit` in CI.
- **[MAJOR] Container runs as root.** No `USER` directive in Dockerfile.
- **[MAJOR] No CSRF protection** on state-changing forms.
- **[MINOR] No brute-force protection** on `/login`.
- **[MINOR] No security.txt / SECURITY.md** responsible-disclosure contact.

**Remediation.** **Rotate every secret today** (Pi SSH, edge API key, admin password, Gemini key — the git history assumes compromise). Delete hard-coded secrets from `INSTRUCTIONS.md`, add an `INSTRUCTIONS.example.md`. Force a generated `SECRET_KEY` on first run or refuse to boot. Remove the admin-password-as-Bearer path (`smartwaste/web.py:96`). Add Caddy or nginx reverse proxy with Let's Encrypt / internal CA. Add non-root `USER` + `HEALTHCHECK` to Dockerfile. Enable Dependabot + CodeQL on GitHub. Target **SOC 2 Type I** within 12 months if enterprise is the plan.

**Effort.** 1 week for the bleeding (rotate + TLS + non-root + default fixes); 2–3 months for SOC 2 readiness.

---

### 4.6 DevOps, Infrastructure & SRE

**What exists today.** Three Docker Compose files, GitHub Actions CI (ruff, mypy, pytest). PostgreSQL healthcheck in compose. Python `logging` to file (`smartwaste/log_setup.py`). Grafana provisioning under `grafana/`.

**Gaps vs. enterprise.**
- **[BLOCKER] Single point of failure everywhere.** One Postgres instance, one FastAPI instance, one Gemini dependency, one laptop at `10.19.189.171` per `INSTRUCTIONS.md:5`. Any restart = outage.
- **[BLOCKER] No backups, no PITR.** `pgdata:/` named volume in `docker-compose.yml:15` — no dump schedule, no off-host copy, no restore test documented.
- **[MAJOR] No `/health` or `/ready` endpoint.** Kubernetes/ECS cannot probe liveness.
- **[MAJOR] No Prometheus metrics / OpenTelemetry traces.** Grafana reads Postgres only; no RED/USE metrics, no latency histograms, no cross-service trace IDs.
- **[MAJOR] No log aggregation.** `logs/run_*.log` is a local file. No Loki / ELK / Datadog / Sentry.
- **[MAJOR] No alerting rules.** Grafana ships dashboards (`grafana/dashboards/`) but no alert policies; silence is the only feedback a down bin gives.
- **[MAJOR] App container has no healthcheck** and no resource limits in `docker-compose.yml:24-40`.
- **[MAJOR] CI is lint + test only.** No coverage gate, no container build, no image signing, no SBOM, no CD, no environments (stage / prod).
- **[MAJOR] No Kubernetes manifests / Helm chart.** Procurement teams that mandate GKE/EKS are blocked.
- **[MINOR] Bind-mount `./smartwaste:/app/smartwaste`** in `docker-compose.yml:40` — useful for dev, risky for prod (host edits land live).

**Remediation.** Add `/health` + `/ready` returning DB + Gemini + disk status. Expose Prometheus metrics via `starlette-prometheus`; ship an OpenTelemetry collector. Pipe logs to Loki. Replace Grafana-only with Grafana + Alertmanager (PagerDuty / Slack). Write a Helm chart; CI builds and pushes signed container images (`cosign`) to GHCR on main. Schedule nightly `pg_dump` → S3, document a verified restore script. Add a `USER`, `HEALTHCHECK`, and `--read-only` to the app image.

**Effort.** 3–4 weeks.

---

### 4.7 Testing & Quality

**What exists today.** Roughly fourteen `tests/test_*.py` files exercising camera, classifier, config, database, dataset, presence, prompt, state, strategies, UI, utils, web. Pytest + coverage configured in `pyproject.toml`. CI runs ruff + mypy + pytest on 3.11.

**Gaps vs. enterprise.**
- **[MAJOR] No end-to-end tests.** Edge-device-to-server-to-dashboard flow is uncovered.
- **[MAJOR] No load or stress tests.** No idea what 100 bins × 1 classification/min does to the system.
- **[MAJOR] No chaos / fault injection.** No Gemini-down, Postgres-down, or network-partition scenarios.
- **[MAJOR] Coverage is not enforced.** No threshold gate in CI; no badge; no per-PR coverage delta.
- **[MAJOR] No security tests.** No OWASP Top 10 fuzzing, no dependency audit, no auth-bypass regression tests.
- **[MAJOR] Postgres path untested in CI.** Tests default to SQLite; production runs Postgres → divergence hides bugs.
- **[MINOR] No property-based tests.** Hypothesis would catch JSON-parsing edge cases in `classifier._extract_json`.
- **[MINOR] No hardware-in-loop.** Real OAK USB disconnects aren't simulated.

**Remediation.** Add a `tests/e2e/` suite using `testcontainers-python` (spin Postgres + FastAPI + a mock Gemini). Run it in CI. Add a `locust` load scenario. Set `--cov-fail-under=70` and ratchet up. Enable CodeQL + `pip-audit`. Add a Postgres CI matrix job.

**Effort.** 2–3 weeks.

---

### 4.8 Frontend / UX / Accessibility

**What exists today.** Four Jinja2 templates (`dashboard.html`, `index.html`, `login.html`, `site.html`) and two stylesheets (`smartwaste/web_static/style.css`, `site.css`) plus `site.js`. Marketing site is genuinely good — glassmorphism, responsive, Leaflet map with Yerevan deployment pins, animated counters.

**Gaps vs. enterprise.**
- **[BLOCKER] Dashboard has three numeric KPIs and no charts.** `smartwaste/web_templates/dashboard.html:130-143` shows `Active Bins / Online Now / Total Classifications` and nothing else. No timeline, no heatmap, no filter, no date range, no drill-down, no export. Compology ships a chart the first time you open the page.
- **[BLOCKER] No map in the operator dashboard.** Marketing site has one; operators do not.
- **[MAJOR] No pagination.** The bins grid `innerHTML = ''` and re-renders every poll (`dashboard.html:173-179`) — browser will stall past ~200 bins.
- **[MAJOR] No alerts UI.** No "bin full," "bin offline >1 h," "Gemini quota exhausted" notifications.
- **[MAJOR] Accessibility fails WCAG 2.1 AA.** No `aria-live` on polled KPIs (`dashboard.html:132-142`), no `:focus-visible` rings, no `prefers-reduced-motion` guard, decorative SVGs missing `aria-hidden`, no skip-link.
- **[MAJOR] Mobile is rough.** The per-bin table (`index.html:75-85`) truncates with `white-space: nowrap; max-width: 120px`; sidebar stacking below a video feed is unreadable on a phone.
- **[MINOR] Inline `<style>` block** in `dashboard.html:8-114` duplicates `style.css` — defeats browser caching.
- **[MINOR] No favicon, no Open Graph tags** on `site.html`.
- **[MINOR] "Coming Soon" placeholders** in `site.html:265-288` read as unfinished to a buyer.
- **[MINOR] No dark/light toggle.** Dark only.

**Remediation.** Ship `Chart.js` or `ECharts`; add a time-range picker component; add a real Leaflet view to `/dashboard`; add a toast/alert drawer. Introduce a React or Svelte build step (`vite`) — vanilla DOM updates won't scale. Audit with `axe-core` in CI. Add a mobile-first card layout replacing the entries table below 768 px. Replace marketing placeholders with embedded Loom/YouTube or hide the section until ready.

**Effort.** 4–6 weeks (a dashboard rewrite is the single largest frontend investment).

---

### 4.9 Internationalization & Localization

**What exists today.** All UI copy is English. Armenian brand names are baked into the Gemini prompt (`smartwaste/prompt.py:22`). `site.html:321-323` lists a `+374` phone and `@smartbin.am` email on an English page.

**Gaps vs. enterprise.**
- **[BLOCKER for Armenian deployment] No Armenian or Russian UI.** The product is deployed in Yerevan (`CLAUDE.md`, `smartwaste/settings.py:49` → `location: "Yerevan"`) with an English-only operator UI. Staff training cost goes up, adoption goes down.
- **[MAJOR] No locale-aware dates/numbers/units.** ISO timestamps rendered raw (`index.html:197`); no 24-hour / DD.MM.YYYY; no kg vs lb toggle.
- **[MAJOR] Brand list hard-coded in prompt.** Every new market requires a code change and a Gemini cost evaluation.
- **[MINOR] No RTL readiness** for future Arabic / Hebrew markets.

**Remediation.** Extract all user-facing strings to `locale/<lang>.json`; integrate `babel` on the backend and `i18next` on any future SPA. Translate first to `hy-AM` and `ru-RU`. Move the brand list to `config/brands/<region>.yaml` loaded into the prompt at runtime.

**Effort.** 2 weeks (strings + hy-AM) + ongoing translation cost.

---

### 4.10 Product, Business & Go-to-Market

**What exists today.** A working demo, a marketing site, one deployed bin in Yerevan.

**Gaps vs. enterprise.**
- **[BLOCKER] No pricing model exposed anywhere.** Competitors publish per-bin-per-month or per-tonnage rates; SmartBin has none visible in `site.html` or docs.
- **[BLOCKER] No customer portal / onboarding flow.** No "sign up → create org → add first bin → invite users" path. Sales teams cannot self-serve demos.
- **[MAJOR] No mobile app.** Field ops live on phones. Bigbelly, Nordsense, Enevo all ship iOS + Android; a PWA would be the minimum credible answer.
- **[MAJOR] No route-optimization module.** Enevo's entire business.
- **[MAJOR] No ESG / carbon-credit reporting.** Municipal RFPs increasingly require it.
- **[MAJOR] No municipal ERP / GIS integrations.** City CRMs (Salesforce Public Sector, Cityworks, ArcGIS).
- **[MAJOR] No SLA tracking.** No per-bin uptime %, missed-pickup counts, service credits.
- **[MAJOR] No SECURITY.md / compliance posture page.** Enterprise procurement sends a vendor-security-questionnaire as the first step; SmartBin cannot fill it out.
- **[MINOR] No case studies / social proof** on `site.html`. "Deployed in Yerevan" with a map is the entire evidence.
- **[MINOR] No developer docs** (`/docs`, SDK, webhook reference).

**Remediation.** Publish a pricing page (even "Contact us" tiers). Build a tenant onboarding wizard. Pick **one** of {route optimization, ESG reporting, municipal ERP} as a first real differentiator — don't try to ship all three. Publish a `SECURITY.md`, `LICENSE`, `CONTRIBUTING.md`. Add a PWA wrapper over the operator dashboard for "mobile app in a week."

**Effort.** Ongoing product work; 6–8 weeks to get the basics visible.

---

## 5. Top 10 Blockers (Prioritized)

| # | Blocker | Severity | Fix window |
|---|---|---|---|
| 1 | Secrets committed in `INSTRUCTIONS.md`; rotate everything | BLOCKER | 1 day (rotate) + 1 week (remove + rewrite history) |
| 2 | Admin password doubles as Bearer token (`smartwaste/web.py:96`) | BLOCKER | 1 day |
| 3 | Default credentials + default session secret shipped | BLOCKER | 1 day |
| 4 | No TLS on any endpoint | BLOCKER | 3 days (Caddy reverse proxy) |
| 5 | No multi-tenancy (single admin, no org model) | BLOCKER | 3 weeks |
| 6 | Simulated sensor data in analytics pipeline | BLOCKER | 2 weeks (remove + real sensors) |
| 7 | Gemini single-vendor lock-in, no fallback | BLOCKER | 4 weeks |
| 8 | No CSV/PDF export from dashboard | BLOCKER | 1 week |
| 9 | No backups / PITR for Postgres | BLOCKER | 3 days |
| 10 | Dashboard has zero charts | BLOCKER | 2 weeks |

---

## 6. 90-Day Remediation Roadmap

### Phase 1 — Stop the bleeding (Days 1–14)
- Rotate every committed secret; purge from git history (BFG / `git filter-repo`); add `SECURITY.md`.
- Delete the admin-password-as-Bearer branch in `smartwaste/web.py:96`.
- Force-generate `SECRET_KEY` on first boot if default is detected; refuse to start with default admin password outside dev.
- Put Caddy in front of FastAPI; terminate TLS with internal CA or Let's Encrypt.
- Add non-root `USER`, `HEALTHCHECK`, resource limits to `Dockerfile` / `docker-compose.yml`.
- Enable Dependabot + CodeQL + `pip-audit` in CI.
- Schedule nightly `pg_dump` to off-host S3-compatible store; test a restore.
- Add `/health` + `/ready` endpoints.
- Delete simulated sensor columns, or rename them `*_demo` and exclude from real dashboards.

### Phase 2 — Parity with single-tenant competitors (Days 15–45)
- Introduce Alembic migrations.
- Schema: `organizations`, `users`, `roles`, `api_keys`, `device_groups`, `audit_log`; add `org_id` FKs; enable Postgres RLS.
- Replace session auth with per-user Argon2 hashes + per-device API keys; add password reset + MFA (TOTP).
- Add `/api/v1/` prefix; deprecate unversioned routes.
- Rewrite dashboard: `vite` + React (or Svelte), Chart.js, time-range picker, map, alerts drawer, CSV + PDF export.
- Add real fill-level: VL53L1X ToF + HX711 load cell on the Pi; populate `weight_g` / `fill_pct`; retire `random.uniform()` (`smartwaste/dataset.py:15-22`).
- Ship i18n scaffolding; translate to `hy-AM` + `ru-RU`.
- Add `slowapi` rate limits + CSRF tokens on state-changing forms.

### Phase 3 — Competitive differentiation (Days 46–90)
- Ship an on-device classifier (YOLOv8-n or fine-tuned MobileNet-SSD on the saved dataset) as a Gemini fallback.
- Return `confidence` and `contamination_pct` in every result.
- OTA: adopt balena / Mender; per-device X.509 certs; remote reboot + diagnostics endpoints.
- Observability: Prometheus metrics via `starlette-prometheus`, Loki for logs, Alertmanager → Slack/PagerDuty, OpenTelemetry traces across edge→server.
- Load tests in CI (`locust`, 100 virtual bins); coverage gate at 70 %.
- Publish Helm chart + signed container images (cosign) to GHCR.
- Pick one differentiator and ship an MVP: route optimization *or* ESG carbon reporting *or* municipal ERP connector.

At day 90 SmartBin should be able to respond credibly to a municipal or hauler RFP.

---

## 7. Competitive Positioning Verdict

**Where SmartBin wins today.** Open-source stack, commodity BOM (OAK-D + Raspberry Pi ~ $400 vs Bigbelly's ~ $4k), genuinely clever on-device sensor fusion (`smartwaste/oak_native.py`), and a brand/visual identity that looks more expensive than the product is. A university, a small hauler, or a single municipality pilot is a realistic sale **today**.

**Where SmartBin loses today.** Every enterprise checklist item — SOC 2, multi-tenant RBAC, OTA, SLA, real fill sensors, on-prem option, mobile app, integrations, certifications. A city procurement officer who asks "can I get a CSV of last month's collections per district, broken down by contamination rate?" gets no for every clause.

**Honest Series-A readiness.** The **narrative** is ready. The **product** is not. An investor who does thirty minutes of diligence (clones the repo, reads `INSTRUCTIONS.md`, greps for `random.uniform`) will flag items 1, 2, 3, and 6 above within a first meeting. Fix those four before the next pitch and the conversation changes entirely.

**The single highest-leverage change.** **Add real weight + fill-level sensors.** Everything downstream — analytics, forecasting, route optimization, ESG reporting, municipal sales — is gated by having one real number that is not `random.uniform()`. Nothing else on this list buys as much credibility per week of engineering.

---

## 8. Appendix — File Map

| File | Why it matters |
|---|---|
| `smartwaste/web.py` | FastAPI app, auth (`:89-98`), in-memory bin registry (`:113`), endpoints. |
| `smartwaste/classifier.py` | Gemini client (`:27,58`), circuit breaker (`:67-100`), retries (`:132-142`). |
| `smartwaste/prompt.py` | Hard-coded classification prompt + Armenian brand list (`:22`). |
| `smartwaste/schemas.py` | `EdgeReport` (unused `weight`, `simulated_*`). |
| `smartwaste/dataset.py` | `random.uniform()` env data (`:15-22`); blank `weight` on save (`:37`). |
| `smartwaste/database.py` | Flat `waste_entries` schema (`:27-65`); Postgres pool `maxconn=5` (`:109-117`). |
| `smartwaste/settings.py` | Defaults: `admin/password123` (`:105-106`), `secret_key` (`:107`), `db_password` (`:97`). |
| `smartwaste/oak_native.py` | On-device sensor fusion — the real engineering differentiator. |
| `smartwaste/edge_client.py` | Edge→server HTTP client + heartbeat. |
| `docker-compose.yml` | Postgres + app + Grafana; Grafana default `admin/admin` (`:46`). |
| `INSTRUCTIONS.md` | Leaks Pi SSH password (`:6`) and edge API key + admin creds (`:8, :80`). |
| `smartwaste/web_templates/dashboard.html` | Operator dashboard — only 3 KPIs (`:130-143`), no charts. |
| `smartwaste/web_templates/index.html` | Per-bin view; mobile-hostile table (`:75-85`). |
| `smartwaste/web_templates/site.html` | Marketing site; placeholders (`:265-288`), contact (`:321-323`). |
| `smartwaste/web_static/style.css` | Dashboard styles — no `:focus-visible`, no reduced-motion. |
| `smartwaste/web_static/site.css` | Marketing styles — solid. |
| `CLAUDE.md` | Project internal notes; brand palette (`:158-195`). |

---

*End of review. Next action recommended: execute Phase 1 of Section 6 this week.*
