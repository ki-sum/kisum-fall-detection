# Experiment notes (public edition)

This is a condensed and edited version of the lab notebook kept during the
experiment (July 2026). Product planning, business content, personal details and
site-specific configuration have been removed. Numbers are reproduced as they
were recorded at the time. Everything here comes from **one person, in one
home, with one hardware set**. It is an engineering log, not a validated study.

Contents:

1. [Hardware setup](#1-hardware-setup)
2. [Phase 1: WiFi CSI only](#2-phase-1--wifi-csi-only-2026-07-15--07-17)
3. [Phase 2: adding 60 GHz mmWave](#3-phase-2--adding-60-ghz-mmwave-2026-07-25--07-28)
4. [Phase 3: fusion dashboard](#4-phase-3--fusion-dashboard-2026-07-28--07-30)
5. [Engineering pitfalls (Windows / Python)](#5-engineering-pitfalls-windows--python)
6. [Not done / open questions](#6-not-done--open-questions)

---

## 1. Hardware setup

- **WiFi CSI**: two ESP32-C6 boards. The sender runs Espressif `csi_send`
  (ESP-NOW). The receiver runs a patched `console_test` that streams `CSI_DATA`
  lines over USB serial at 2 Mbaud (see [`../patches/`](../patches/)).
  - In the final setup the two boards were placed horizontally about 1 m apart,
    with the person in between.
  - The posture test in Phase 1 used a *vertical* TX–RX axis instead.
- **Seeed MR60FDA2** (60 GHz fall detection, XIAO ESP32-C6, ESPHome pre-flashed).
- **Seeed MR60BHA2** (60 GHz breathing / heart rate, XIAO ESP32-C6, ESPHome
  pre-flashed).
- Optional USB webcam for the MediaPipe pose heatmap panel.
- The 60 GHz modules and the 2.4 GHz CSI link ran at the same time with no
  observed interference.

---

## 2. Phase 1 — WiFi CSI only (2026-07-15 … 07-17)

### Signal pipeline

1. Per-subcarrier CSI amplitude (≤ 56 subcarriers, ~100 packets/s).
2. Every 0.2 s, take a 2 s Hanning-windowed FFT per subcarrier and average the
   spectra across subcarriers.
3. Sum the spectrum into four Doppler bands:

| Band | Range | Intended meaning |
|---|---|---|
| breath | 0.1–0.6 Hz | breathing, slow sway |
| slow | 0.6–2 Hz | small adjustments |
| walk | 2–6 Hz | walking, arm swing |
| fall | 6–25 Hz | sudden motion (fall, jump, sit-down) |

### FallDetectorV2 (rule-based)

A state machine on `walk + fall` energy:

1. **Burst**: `walk+fall` rises above `BURST_MIN`.
2. **Peak check**: the burst peak must stay below `BURST_PEAK_MAX`. A larger
   peak is treated as walking, not falling.
3. **Stillness**: after the burst, energy must drop below `STILLNESS_MAX`.
4. **Fire**: if the stillness range stays inside `[RANGE_MIN, RANGE_MAX]` for
   ≥ 6 s, the detector fires. A 15 s cooldown follows.

The first version used fixed absolute thresholds (1200 / 2200 / 900 / 60–300).

**Live testing on 2026-07-17 with fixed thresholds:**

| Scenario | Result |
|---|---|
| Walking around the room | correctly rejected |
| Walking out through the door | correctly rejected (burst peak too large) |
| Empty room | correctly rejected (stillness too flat) |
| Sitting still eating / reading | rejected only **unreliably** (stillness range ~220 falls inside the 60–300 window) |

### Adaptive baseline (V3.1)

The `walk+fall` baseline drifted between sessions. Median `walk+fall` was:

- ≈ 737 in the labeled session
- ≈ 623 in one live session
- ≈ 666 in another

So the same fixed thresholds gave different sensitivities. V3.1 tracks the p10
of `walk+fall` over the last 5 minutes as the room baseline and expresses every
threshold as a ratio of it:

| Threshold | Ratio of baseline |
|---|---|
| `BURST_MIN` | 2.4× |
| `BURST_PEAK_MAX` | 4.4× |
| `STILLNESS_MAX` | 1.8× |
| stillness range | 0.12× … 0.60× |

This version is what both monitors ship.

Labeled recordings of five scenarios (baseline_still, walking, sitting_down,
lying_down_sleep, fall_simulate) are plotted in
[`images/timeseries_all_scenarios.png`](images/timeseries_all_scenarios.png) and
[`images/fall_shape_comparison.png`](images/fall_shape_comparison.png).

### The physical limit

Single-antenna, amplitude-only CSI could not distinguish between:

- a person lying still on the floor after a fall,
- a person lying still in bed,
- a person sitting still (e.g. eating and reading).

Amplitude-only CSI on one receive antenna captures *motion over time*, not
*position* or *pose*. Telling "on the floor" from "on the bed" needs some extra
information, for example:

- phase difference across ≥ 2 receive antennas,
- Doppler direction from a real MIMO array,
- range resolution (e.g. 60 GHz radar),
- context from another sensor.

V2, V3.1 and the various gating rules all hit the same ceiling. A more honest
name for what the CSI detector does is **"vertical motion → horizontal
stillness" transition detector**. It fires on real falls *and* on intentionally
lying down, and it cannot tell them apart.

### Posture hypothesis: disproven (2026-07-17 evening)

**Hypothesis.** With a vertical C6 axis, breathing while lying flat (chest
moving along the axis) should give ≥ 2× the breath-band energy of breathing
while standing.

**Data.** Four 60 s scenarios, median breath band:

| Posture | Median breath band |
|---|---|
| standing_still | 8.3 |
| sitting_still | 3.9 |
| tilted_recline | 4.4 |
| lying_flat | 5.3 |

**Result.** lying / standing = **0.64**, the opposite of the prediction.

**Explanation.** Postural sway while standing (roughly 1–2 cm at 0.2–1 Hz)
falls inside the breath band and is larger than the chest motion from
breathing.

**Consequences:**

- The breath band can roughly separate *standing* from *sitting / lying*.
- It cannot separate *sitting* from *lying*.
- It does not solve "fallen on the floor vs. asleep in bed".

### Effective range

With the V3 stillness indicator, walking more than ~2 m away from the receiver
made the breath band drop below the "moving" threshold (30), even though the
person was still moving. In this setup, one ESP32-C6 pair gave roughly a 2 m
radius for detailed motion. Treat the ESP32-C6 as a prototyping platform, not as
whole-home coverage.

### Stillness indicator

A Schmitt trigger on the walk+fall band was replaced by a single threshold on
the breath band (`breath >= 30 → MOVING`), which felt more responsive in live
use. The stillness panel shows four levels (VERY STILL / STILL / SMALL
MOVEMENTS / LARGE MOVEMENTS) based on the breath value alone. An earlier
standing / sitting / lying classification was removed because it was not
supported by the data (see the posture test above).

### Decision

**Phase 1 decision:** no WiFi-CSI-only fall detector. CSI is kept as a
presence / motion / activity layer, and a dedicated 60 GHz mmWave fall sensor
was added.

---

## 3. Phase 2 — adding 60 GHz mmWave (2026-07-25 … 07-28)

### MR60FDA2 bring-up

- Arrived with **ESPHome 2025.4.0** pre-flashed. No custom firmware was needed.
- **WiFi provisioning:**
  1. On first boot the module opens an AP named `seeedstudio-mr60fda2`.
  2. A captive portal at `http://192.168.4.1` takes the SSID and password.
  3. The credentials persist across reboots.
- **Discovery:** via mDNS (`seeedstudio-mr60fda2-kit-<suffix>.local`) or a scan
  of TCP port 6053.
- The enclosure's USB-C port enumerated as a **CH340** serial port that stayed
  silent at every baud rate tried. It is probably wired to the radar chip's
  configuration UART, not the ESP32 log. All data in this project goes over
  WiFi via the ESPHome native API.
- The ESPHome API was **unauthenticated** (empty password). Any real deployment
  should enable ESPHome API encryption.

**Entities used:**

| Entity | Meaning |
|---|---|
| `person_information` | presence (binary) |
| `falling_information` | fall (binary) |

**Configurable selects:**

| Select | Options |
|---|---|
| `set_install_height` | 2.4 / 2.5 / … / 3.0 m on our unit's ESPHome firmware |
| `set_height_threshold` | 0.0 … 0.6 m |
| `set_sensitivity` | 1 / 2 / 3 |

### Placement findings (MR60FDA2)

| Placement | Observed behaviour |
|---|---|
| A: on a desk, ~60–80 cm above the user's head, pointing down | Presence OK. `falling_information` went True ~5 s after connecting and stayed True ~90 s while the user sat typing (not fallen). Then both sensors went False although the user was still there. |
| B: 2.1 m up, flush against a wall, pointing down | Presence OK. `falling_information` fired **~4–7 s after every person detection**, while walking or standing. Sensitivity 1 did not change this. |
| C: true ceiling mount at 2.5 m, pointing straight down, ≥ 1 m clear floor radius | Usable with the configuration below. |

Configuration used for placement C:

```
install_height    = 2.5 m   (matches the actual mount)
height_threshold  = 0.5 m   (default)
sensitivity       = 1       (at 2, squatting to pick something up triggered FALL)
```

- At sensitivity 1, walking and squatting were rejected.
- Simulated falls (standing → floor, staying down ≥ 15 s) triggered
  `falling_information` within roughly 5–15 s.
- The number of trials was not recorded, so no accuracy figure is claimed.

**Interpretation.** Below the supported mount height, or close to walls, the
fall output degraded to roughly "a person is present" without any warning to
the user. `person_information` was reliable in every placement. Two variables
were **not** separated: wall multipath versus mount height alone.

### MR60BHA2 observations

- Entities: `person_information`, `real-time_respiratory_rate`,
  `real-time_heart_rate`, `distance_to_detection_object`, `target_number`.
- It was mounted on a desk 0.6–1 m from the user.
- When the user sat very still (typing, watching the screen), the respiratory
  rate decayed from ~17/min to 1/min over ~30 s. Heart rate held up better.
- `target_number` sometimes reported 2 with one person in the room, likely
  because of strong reflectors (wall, monitor stand).

---

## 4. Phase 3 — fusion dashboard (2026-07-28 … 07-30)

`unified_monitor.py` runs all sources at once.

### Fusion rules (as implemented)

| Signal | Rule |
|---|---|
| Presence (Living Room zone) | MR60BHA2 **or** CSI. MR60FDA2 is excluded because it covers a different zone. |
| Presence / fall (Bedroom zone) | MR60FDA2 only |
| Motion level | CSI only: STILL / LIGHT / ACTIVE / INTENSE at 1.5× / 2.5× / 5.0× the rolling p10 baseline |
| Breath | MR60BHA2 if in [8, 30]/min, else CSI if in range, else "—". An implausible value such as "1/min" is never shown. |
| Heart rate | Plain average of MR60BHA2 and CSI if both in [40, 180] bpm, else whichever one is in range. CSI heart rate is experimental and SNR-gated (often silent). |
| Fall | MR60FDA2 `falling_information`. The CSI FallDetectorV2 is shown for reference only. |

CSI vitals use a separate 10 s window over the mean amplitude across
subcarriers, with parabolic peak interpolation and EMA smoothing. The motion
analysis keeps its 2 s window.

### Fusion example observed

1. While the user sat still, MR60BHA2's respiratory rate decayed 17 → … → 1/min.
2. The fused output switched to the CSI estimate (shown in amber with source
   "CSI").
3. It did not display the misleading 1/min value.

The CSI breath estimate itself was **not** validated against a reference
measurement (see [section 6](#6-not-done--open-questions)).

### Alerts

- **Fall alert.** An MR60FDA2 False→True transition sends an optional Telegram
  Bot message with inline buttons (Acknowledged / Call user / Call emergency).
- **Button presses.** These come back via long-polling `getUpdates` and appear
  in the on-screen "Family App" preview.
- **Inactivity alert.** Fires when someone is present with no CSI motion for
  `INACTIVITY_ALERT_SEC`. This is 120 s for the demo; a real deployment would use
  a much longer value.

### Known limits documented

1. **Fall latency.** We observed about 5–15 s from the MR60FDA2. That suits
   "fall and call for help", not prevention of injury.
2. **CSI "still-person hole".** A motionless person produces almost no Doppler
   energy, so CSI reports "no person". mmWave presence covers this.
3. **MR60BHA2 breath rate.** Unreliable when the user sits still at a desk.
4. **MR60BHA2 `target_number`.** It can report 2 for one person, so person
   counts from this sensor alone are not trustworthy.
5. **MR60FDA2 sensitivity.** Sensitivity 2 produced false falls on squatting
   in our setup, so 1 was used.
6. **MR60FDA2 mount height.** Mounting below the supported height made the
   fall output unusable.
7. **Vibration.** Continuous vibration (e.g. construction next door) was
   identified as a likely failure mode for both CSI and mmWave. It was **not**
   tested.

### Baseline percentile

p10 (sensitive to environment shifts) was compared with p25 (more stable). p10
was kept. The trade-off is that ACTIVE labels dominate when the noise floor
rises.

### Privacy note

Without images, CSI and mmWave data still describe a person's behaviour.
Heart-rate and breathing data are health data in many jurisdictions (e.g.
GDPR Art. 9). Keep raw data on the device, transmit only events, and get
explicit consent before recording anyone.

---

## 5. Engineering pitfalls (Windows / Python)

- **`cv2.VideoCapture(..., CAP_DSHOW)` can pin the process after exit.** A
  pending DirectShow I/O request kept the process kernel object alive, so the
  parent terminal never got its prompt back. The cause was found in about a
  minute with a `DISABLE_POSE=True` isolation test, after two hours of guessing.
  Fixes:
  - use `CAP_MSMF` (slow to open, 10–30 s, but releases cleanly),
  - call `PoseBackend.force_release_camera()` from the main thread in
    `_on_close()`,
  - keep an external PowerShell `Process.Kill()` as a safety net.
- **Tk pump must reschedule in `finally`.** A single `NameError` in an event
  handler once stopped `_pump()`, and the UI looked frozen while all backends
  were connected. Each event handler is now isolated with its own
  `try/except`.
- **Tk `PhotoImage` lifecycle.** Calling `config(text=...)` on a label whose
  image was replaced in the same tick raised `TclError: image "pyimageN" doesn't
  exist`. Use a separate label for overlay text.
- **MediaPipe VIDEO-mode tracker can get stuck after fast motion.** The
  landmarker is recreated if there are no landmarks for 8 s while frames are
  still arriving.
- **Opening the C6 serial port with `dtr=False, rts=False`.** This toggled the
  auto-reset circuit and left a board in bootloader mode; only a USB re-plug
  fixed it. Open the port without touching DTR/RTS.
- **LAN scans are ambiguous with several ESPHome devices.** Verify
  `device_info().name` against the expected hostname and quarantine
  mismatching IPs. This is implemented.
- **Chrome `--window-position` is honoured only on the first launch of a
  `--user-data-dir` profile.** Query window rectangles at use time.
- **VSCode integrated terminal.** It behaved differently from a standalone
  terminal when a GUI child process terminated itself. Prefer standalone
  Windows Terminal / PowerShell for the Tk app.

---

## 6. Not done / open questions

These were planned but **not** carried out, so the repository contains no
results for them:

- A multi-hour or 24 h log at rest (false-positive rate of a correctly mounted
  MR60FDA2, MR60BHA2 dropout frequency).
- Validating the CSI breath-rate estimate against a manual count or a reference
  device.
- A two-person scenario (does `target_number` work, does the fall output still
  work?).
- Separating wall multipath from mount height in the MR60FDA2 placement
  findings.
- CSI ideas that were never implemented:
  - aggregate-amplitude presence check to cancel alerts when the room is empty,
  - two independent receiver pairs for spatial diversity,
  - a small supervised classifier trained on 20–30 labeled sessions across
    rooms and people.
