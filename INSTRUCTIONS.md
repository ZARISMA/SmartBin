# SmartBin — Bring-up Instructions

Two machines:

- **Laptop** (Windows, Docker Desktop) — runs the server (FastAPI + Postgres + Grafana) at `http://10.19.189.171:8000`.
- **Raspberry Pi** (`hexabin@10.19.189.238`, pass `Hexa1234`) — edge device, reports to the laptop.

Shared edge API key: `smartbin-edge-2026-a7f3k9`. Admin login: `admin` / `password123`.

---

## 1. Laptop — start the server

Open git-bash (or PowerShell) in the repo root and run:

```bash
in /SmartBin
docker compose up -d
```

Check it's healthy:

```bash
docker compose ps
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/
```

Open `http://localhost:8000` in a browser. Log in with `admin` / ``.

**If the app container was rebuilt (code/dep changes to `requirements.txt` or `Dockerfile`):**

```bash
docker compose up -d --build
```

Code under `smartwaste/` is bind-mounted — for pure Python edits, `docker compose restart app` is enough.

**If port 5432 is already in use on the host** (another Postgres): already handled — `docker-compose.yml` maps Postgres to host port `5433`. Ignore.

**Windows Firewall** (one-time, elevated PowerShell — only needed if the Pi can't reach the laptop):

```powershell
New-NetFirewallRule -DisplayName "SmartBin 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

---

## 2. Raspberry Pi — start the edge

SSH in from the laptop (git-bash):

```bash
plink -ssh -batch -pw "Hexa1234" -hostkey "SHA256:5kgrm96PgRykScn0pfXTzMBMmKP8spqr0T6pr/2aE9I" hexabin@10.19.189.238
```

Or use OpenSSH: `ssh hexabin@10.19.189.238` (password `Hexa1234`).

### 2a. With an OAK camera plugged in (normal operation)

```bash
cd ~/SmartBin
docker compose -f docker-compose.edge.yml up -d
docker compose -f docker-compose.edge.yml logs -f --tail 50
```

Ctrl-C detaches from logs; container keeps running (`restart: unless-stopped`).

### 2b. No camera connected (connection test only)

`mainoak.py` needs the OAK camera and will crash-loop without one. Run a pure-heartbeat process instead so the bin shows Online on the dashboard:

```bash
cd ~/SmartBin
docker compose -f docker-compose.edge.yml down 2>/dev/null

cat > /tmp/hb.sh <<'EOF'
#!/bin/bash
while true; do
  curl -s -o /dev/null -X POST http://10.19.189.171:8000/api/heartbeat \
    -H "Authorization: Bearer smartbin-edge-2026-a7f3k9" \
    -H "Content-Type: application/json" \
    -d '{"bin_id":"bin-01","status":"online","camera_mode":"none","uptime_seconds":0}'
  sleep 30
done
EOF
chmod +x /tmp/hb.sh
pkill -f /tmp/hb.sh 2>/dev/null
setsid bash -c '/tmp/hb.sh >/tmp/hb.log 2>&1 </dev/null &'
pgrep -af hb.sh
```

To stop the heartbeat loop later:

```bash
pkill -f /tmp/hb.sh
```

---

## 3. Verify end-to-end

From the laptop:

```bash
curl -s -H "Authorization: Bearer smartbin-edge-2026-a7f3k9" http://localhost:8000/api/dashboard
```

Expect `bin-01` with `"status":"online"`. In the browser, the dashboard card will update within 5 s (poll interval). A bin flips to offline 60 s after its last heartbeat.

---

## 4. Shut down

**Laptop:** `docker compose down` (add `-v` to wipe Postgres data).
**Pi (camera mode):** `docker compose -f docker-compose.edge.yml down`.
**Pi (heartbeat-only mode):** `pkill -f /tmp/hb.sh`.

---

## Key config reference

| Setting | Laptop (.env) | Pi (.env) |
|---|---|---|
| `SMARTWASTE_EDGE_MODE` | `false` | `true` |
| `SMARTWASTE_BIN_ID` | `server-01` | `bin-01` |
| `SMARTWASTE_SERVER_URL` | — | `http://10.19.189.171:8000` |
| `SMARTWASTE_EDGE_API_KEY` | `smartbin-edge-2026-a7f3k9` | same |
| `SMARTWASTE_CAMERA_MODE` | `none` (docker-compose.yml overrides) | `oak-native` with camera, `none` without |
| `SMARTWASTE_DB_BACKEND` | `postgresql` (docker-compose.yml overrides) | n/a |

Laptop IP `10.19.189.171` and Pi IP `10.19.189.238` are LAN addresses — re-check with `ipconfig` / `ip a` if either machine gets a new DHCP lease.
