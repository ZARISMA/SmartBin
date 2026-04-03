# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart Waste AI — a real-time waste classification system using dual OAK-D USB3 depth cameras and Google Gemini vision API. It captures side-by-side frames from two cameras, classifies waste into 7 categories, and logs results to JSON/Excel/image files.

**Primary entry point:** `main.py`

## Environment Setup

**Python requirement: 3.10+** (3.11 recommended; ships by default on Raspberry Pi OS Bookworm)

```bash
# Activate virtual environment
# Windows:
oak_env\Scripts\activate
# Linux / Raspberry Pi:
source oak_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set required API key
# Windows (PowerShell):
$env:GEMINI_API_KEY='your_key_here'
# Linux / Raspberry Pi:
export GEMINI_API_KEY='your_key_here'
```

### Linux / Raspberry Pi — one-time setup

**1. USB permissions (udev rules)** — required or cameras won't be detected:
```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Reconnect the cameras after running this.

**2. Display** — a monitor and desktop session must be active. When using SSH:
```bash
export DISPLAY=:0
python main.py
```

**3. OpenCV on ARM** — if `pip install opencv-python` fails, use the system package instead:
```bash
sudo apt install python3-opencv
```
Then remove `opencv-python` from `requirements.txt`.

## Running the Application

**OAK cameras (default):**
```bash
python main.py
```

**Raspberry Pi cameras:**
```bash
python mainraspberry.py
```

**Runtime controls:**
- `c` — manually classify current frame
- `a` — toggle auto-classify mode (every 6 seconds)
- `q` — quit

## Hardware Requirements

- 2x OAK-D USB3 cameras (uses `depthai` SDK) — for `main.py` / `mainauto.py`
- **OR** 2x Raspberry Pi cameras (uses `picamera2`) — for `mainraspberry.py`
- Internet access for Gemini API calls

### Raspberry Pi camera setup

```bash
# Install picamera2 (already available on Pi OS Bookworm)
sudo apt install -y python3-picamera2

# Or inside a venv:
pip install picamera2

# Enable cameras in /boot/firmware/config.txt:
# Pi 5 dual native ports — default (camera_auto_detect=1 is already on)
# Camera multiplexer:
#   dtoverlay=camera-mux-4port
```

## Key Configuration Constants (in `smartwaste/config.py`)

| Constant | Value | Purpose |
|---|---|---|
| `VALID_CLASSES` | 7 categories | Plastic, Glass, Paper, Organic, Aluminum, Other, Empty |
| `AUTO_INTERVAL` | 6 seconds | Delay between auto-classifications |
| `CROP_PERCENT` | 0.20 | Crops 20% from sides before AI analysis |
| `MAX_DT` | 0.25 sec | Max frame timestamp delta between cameras |
| `MODEL_NAME` | `gemini-3-flash-preview` | Gemini model used |
| `JPEG_QUALITY` | 85 | Compression for API image uploads |

## Architecture

**Threading model:**
- Main thread handles video capture, UI rendering, and keyboard input
- Daemon worker threads handle Gemini API calls asynchronously
- Global flags `is_classifying` and `AUTO_CLASSIFY` coordinate state

**Data flow:**
1. Dual camera frames captured independently → synced by timestamp (`MAX_DT`)
2. Frames concatenated horizontally → cropped → JPEG-encoded
3. Bytes sent to Gemini with a structured prompt → strict JSON response parsed
4. Result logged to: console, `logs/run_*.log`, `waste_dataset/metadata.json`, `waste_dataset/waste_log.xlsx`, and `waste_dataset/*.jpg`

**AI prompt** (`smartwaste/prompt.py`): Instructs Gemini to return `{"category": ..., "description": ..., "brand_product": ...}`. Includes Armenian brand examples (Jermuk, BOOM, etc.) relevant to the deployment location (Yerevan).

## Data Storage

- `waste_dataset/metadata.json` — accumulated classification records
- `waste_dataset/waste_log.xlsx` — Excel log with simulated environmental data (temperature, humidity, vibration, air pollution, smoke)
- `waste_dataset/*.jpg` — captured frames, timestamp-named
- `logs/` — per-run log files and API error dumps

## Module Structure

```
main.py                  ← entry point (OAK cameras)
mainauto.py              ← entry point (OAK cameras, auto gate mode)
mainraspberry.py         ← entry point (Raspberry Pi cameras)
smartwaste/
  __init__.py
  config.py              ← all constants and paths
  cameraOak.py           ← OAK pipeline setup and frame cropping
  cameraraspberry.py     ← Raspberry Pi picamera2 setup and frame capture
  classifier.py          ← Gemini API call, JSON parsing
  dataset.py             ← save images, metadata.json, Excel log
  log_setup.py           ← logging configuration
  prompt.py              ← Gemini prompt string
  state.py               ← thread-safe AppState class
  ui.py                  ← OpenCV overlay rendering
```

## Testing

When adding new functionality or modifying existing behaviour, add corresponding tests in `tests/` if the change introduces new logic worth verifying.

```bash
# Run the full suite
python -m pytest tests/

# With coverage report
python -m pytest tests/ --cov=smartwaste --cov-report=term-missing
```

Test files follow the pattern `tests/test_<module>.py`. Install test dependencies with `pip install -r requirements-test.txt`.

## Known Issues

1. Gemini quota errors (HTTP 429) are handled but may silently skip classifications
2. OAK-D devices are auto-detected by index — swap physical USB ports if wrong camera is left/right
