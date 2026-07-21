# 2. Camera geometry: server desired-state vs bin-local applied-state

Date: 2026-07-21

## Status

Accepted

## Context

Operators need to rotate, flip and crop each of a bin's two cameras from the
dashboard, and have the settings persist. This is the first per-device,
runtime-editable setting in the system.

The complication is **who owns the cameras**. In the unified deployment the
cameras may run in the same process as the dashboard (local `python -m
hexabin.web`) or on a remote edge bin (Pi) that the dashboard only reaches over
the existing command/heartbeat channel. Config that lives only on the server
can't be read by a remote capture loop; config that lives only on the bin can't
be shown on the dashboard when the bin is offline.

The old global `CROP_PERCENT` was immutable env config applied identically to
both cameras — none of this machinery existed.

## Decision

Split the state by role:

- **Server DB (`camera_configs` table) = fleet desired-state.** Every save
  upserts here (keyed by `bin_id` + `cam_index`). This is what the dashboard
  reads and what survives even when a bin is offline.
- **Bin-local `camera_config.json` = applied-state.** The process that owns the
  cameras seeds an in-memory `CameraConfigStore` from this file at startup and
  reads it every capture iteration, so an edit applies live and survives a
  restart without one.

A save reaches the cameras by the same path a bin is otherwise controlled: for a
local bin the server writes the JSON + updates the in-memory store directly; for
a remote bin it sends a `set_camera_config` command over the existing
sidecar `/command` channel, and the bin writes its own JSON.

The crop is stored **normalized** `(x0, y0, x1, y1)` in `[0,1]` of the
*rotated+flipped* frame; transform order is fixed at **rotate → flip → crop**.
The default (no saved config) is derived from `CROP_PERCENT`, so an unedited
camera behaves byte-for-byte as before.

## Consequences

- **Positive:** works identically whether cameras are local or remote; edits
  apply live (no restart); backward compatible until a camera is actually edited.
- **Positive:** normalized crop is resolution-independent and round-trips cleanly
  between the browser editor's canvas and the server's `apply_transform`.
- **Negative / surprising:** the two stores can diverge. Saving to an **offline**
  bin updates the desired-state DB but not the bin; it reconciles only when the
  operator saves again while the bin is online. There is deliberately **no
  automatic push on reconnect** yet — a future enhancement.
- **Scope limit:** the dual-OAK pipeline (`web.py` loops + `app.py` run loop) and
  the edge sidecar are fully wired. The `mainoak.py` OAK-native *standalone* loop
  still uses the legacy crop; its sidecar keeps working (the store param is
  optional) but does not apply live edits. Acceptable because the editor targets
  the two-camera pipeline.
