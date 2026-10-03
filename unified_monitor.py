"""
unified_monitor.py — Tk dashboard fusing MR60FDA2 (fall) + MR60BHA2 (vitals) + CSI.

Two zone tabs plus an event log:
  Living Room  — [mmWave raw] | [CSI raw] | [Assessment & Analysis (fused)]
                 plus CSI motion plot and heart-rate history plot
  Bedroom      — MR60FDA2 presence + fall panels only
  Events       — event log
Left column: optional webcam pose heatmap + "Family App" alert preview.

Fusion rules (Living Room zone uses MR60BHA2 + CSI only):
  Presence:   MR60BHA2 or CSI says PRESENT → PRESENT
  Motion:     CSI only (STILL / LIGHT / ACTIVE / INTENSE vs. rolling baseline)
  Breath:     MR60BHA2 primary; CSI fallback if MR60BHA2 outside [8, 30]/min;
              "—" if neither is in range
  Heart:      plain average of MR60BHA2 + CSI when both in [40, 180] bpm,
              otherwise whichever one is in range (CSI heart rate is experimental)
  Fall:       MR60FDA2 falling_information is authoritative; the CSI
              FallDetectorV2 runs for reference only

Toggles: "record CSV log" (1 Hz rows to unified_logs/*.csv), "● REC" (MP4 of
the monitor window to demo_recordings/).

The CSI serial port is exclusive — close legacy/kisum_radar_monitor.py first.

Run: python unified_monitor.py --help
"""

import argparse
import asyncio
import base64
import concurrent.futures
import csv
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import scrolledtext, ttk

import cv2
import mss
import numpy as np
import serial
from PIL import Image, ImageTk

from aioesphomeapi import APIClient

# Reuse pose processing from pose_stickfigure.py (no need to duplicate)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mediapipe as mp
from pose_stickfigure import (
    ensure_model, create_pose_landmarker, apply_anonymization,
    draw_skeleton, LandmarkSmoother, PALETTES, COLOUR_SILHOUETTE_DEFAULT,
)

# =============================================================================
# CONFIG
# =============================================================================

# All site-specific values (mDNS hostnames, IPs, serial port, Telegram
# credentials) come from command-line flags or environment variables — see
# main(). Nothing device-specific is hardcoded here.
#
# mdns:        the module's ESPHome hostname, e.g.
#              "seeedstudio-mr60fda2-kit-xxxxxx.local" (suffix differs per unit).
#              Discovery order: mDNS → fallback_ip → LAN scan with identity
#              check against this hostname.
# fallback_ip: optional speed-up when mDNS is unreliable on your network.
DEVICES = [
    {"label": "FDA2",
     "mdns": os.environ.get("FDA2_HOST", ""),
     "fallback_ip": os.environ.get("FDA2_IP") or None},
    {"label": "BHA2",
     "mdns": os.environ.get("BHA2_HOST", ""),
     "fallback_ip": os.environ.get("BHA2_IP") or None},
]
API_PORT = 6053
LAN_PREFIX = os.environ.get("LAN_PREFIX", "192.168.1.")   # /24 scanned as last resort

CSI_PORT = os.environ.get("CSI_PORT", "")   # e.g. "COM5" on Windows
CSI_BAUD = 2000000

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unified_logs")
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_recordings")

BHA2_BREATH_MIN = 8
BHA2_BREATH_MAX = 30
BHA2_HEART_MIN = 40
BHA2_HEART_MAX = 180

# Public-facing display names (used in UI; internal state keys stay FDA2/BHA2/CSI/POSE)
APP_TITLE = "Ambient Care  ·  KI SUM AI"
DISPLAY_NAMES = {
    "FDA2": "mmWave Bedroom",
    "BHA2": "mmWave Living Room",
    "CSI":  "CSI (WiFi)",
    "POSE": "Camera",
}
# Compact tags used in narrow columns (e.g., signal row source labels)
COMPACT_NAMES = {
    "FDA2": "Bedroom",
    "BHA2": "Living Room",
    "CSI":  "CSI",
    "POSE": "Camera",
}
# Human-readable labels for event log (never expose vendor codes like BHA2)
LOG_LABEL_NAMES = {
    "FDA2": "mmWave Fall",
    "BHA2": "mmWave Vitals",
    "CSI":  "CSI Motion",
    "POSE": "Camera",
    "SYS":  "System",
}

PLOT_WINDOW_SEC = 30.0
HR_PLOT_WINDOW_SEC = 120.0  # 2 min so exercise test (sit-run-sit) fits

# ---- Pose / webcam config ----
# DISABLE_POSE=True (or --no-pose) skips webcam+MediaPipe entirely — useful for
# headless CSI+mmWave-only tests. Clean shutdown with the webcam relies on the
# CAP_MSMF backend + explicit early cap.release() in _on_close.
DISABLE_POSE = False
WEBCAM_INDEX = 0
POSE_MODE = "heatmap"       # overlay | blur | silhouette | skeleton | heatmap
POSE_PALETTE = "viridis"
POSE_SMOOTH = 0.5
POSE_DISPLAY_SIZE = (640, 360)   # size shown inside monitor (aspect 16:9)

# ---- Screen recorder ----
SCREEN_REC_FPS = 15

# ---- Telegram Web (optional Chrome app-mode window, --telegram-web) ----
AUTO_OPEN_TELEGRAM_WEB = False
TELEGRAM_WEB_URL = "https://web.telegram.org/a/"
TELEGRAM_CHROME_W = 540           # phone-ish narrow width
TELEGRAM_CHROME_H = 820
TELEGRAM_CHROME_GAP = 20          # px gap between monitor and Chrome
# Separate Chrome profile so Telegram login persists across sessions AND
# doesn't collide with your normal Chrome browsing profile
TELEGRAM_CHROME_PROFILE = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "KisumMonitorChromeProfile",
)

# ---- Telegram alerts (optional; set env vars to enable) ----
# How to get these (one-time, ~2 min setup):
#   1. Open Telegram on phone, search @BotFather, /newbot, follow prompts.
#      BotFather gives you a token like "1234567890:AAxxxxxxxxxxxxxxxxxx"
#   2. Message YOUR new bot at least once (e.g. "hi")
#   3. In a browser: https://api.telegram.org/bot<TOKEN>/getUpdates
#      Look for "chat":{"id": 123456789, ...} — that number is your chat_id
#   4. Set the env vars TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before launching
#      (cmd: set NAME=value, PowerShell: $env:NAME="value").
# Never commit a real token. Empty = alerts appear only in the on-screen preview.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# Alert timing
FALL_ALERT_COOLDOWN_SEC = 30       # don't spam if fall re-fires within 30s
INACTIVITY_ALERT_SEC    = 120      # send alert if presence + no motion for this long
                                    # (real deployment would use ~3600; 120 for demo)

# =============================================================================
# CSI constants (mirror legacy/kisum_radar_monitor.py — kept in sync manually)
# =============================================================================

DOPPLER_WINDOW_SEC = 2.0
DOPPLER_HOP_SEC = 0.2
CSI_EXPECTED_RATE_HZ = 100.0
DOPPLER_MAX_SUBCARRIERS = 56

DOPPLER_BANDS = [
    ("breath", 0.10, 0.60),
    ("slow", 0.60, 2.00),
    ("walk", 2.00, 6.00),
    ("fall", 6.00, 25.00),
]

CSI_RE = re.compile(
    r"CSI_DATA,\d+,(\d+),\d+,\S*?,[0-9a-fA-F:]+,"
    r"(-?\d+),\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,"
    r"(-?\d+),\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,"
    r"(\d+),\d+,"
    r"([A-Za-z0-9+/=]+)"
)


def parse_csi_line(line):
    m = CSI_RE.search(line)
    if not m:
        return None
    try:
        b64 = m.group(5)
        valid_len = int(m.group(4))
        pad = (-len(b64)) % 4
        raw = base64.b64decode(b64 + "=" * pad, validate=False)
        arr = np.frombuffer(raw, dtype=np.int8)
        if len(arr) < 4:
            return None
        arr = arr[: min(len(arr), valid_len)]
        if len(arr) % 2:
            arr = arr[:-1]
        complex_csi = arr[0::2].astype(np.float32) + 1j * arr[1::2].astype(np.float32)
        amplitude = np.abs(complex_csi)
        if len(amplitude) > DOPPLER_MAX_SUBCARRIERS:
            amplitude = amplitude[:DOPPLER_MAX_SUBCARRIERS]
        return {"ts": datetime.now().timestamp(), "amplitude": amplitude}
    except Exception:
        return None


def _parabolic_peak_hz(powers, freqs, peak_idx):
    """Sub-bin peak location via 3-point parabolic fit around peak_idx."""
    if peak_idx <= 0 or peak_idx >= len(powers) - 1:
        return float(freqs[peak_idx])
    y0, y1, y2 = float(powers[peak_idx - 1]), float(powers[peak_idx]), float(powers[peak_idx + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(freqs[peak_idx])
    delta = 0.5 * (y0 - y2) / denom  # in "bin units", range roughly [-0.5, +0.5]
    delta = max(-0.5, min(0.5, delta))
    if peak_idx + 1 < len(freqs):
        bin_w = float(freqs[peak_idx + 1] - freqs[peak_idx])
    else:
        bin_w = float(freqs[peak_idx] - freqs[peak_idx - 1])
    return float(freqs[peak_idx]) + delta * bin_w


# Vitals analysis config — longer window than motion detection for better freq resolution.
# 10s window at 100 Hz sample rate → freq bin width 0.1 Hz = 6 BPM raw,
# then parabolic interpolation for sub-bin accuracy (~1-2 BPM effective).
VITALS_WINDOW_SEC = 10.0
VITALS_HOP_SEC = 0.5
VITALS_SMOOTH_ALPHA = 0.3  # EMA weight of new sample (0=frozen, 1=raw)


class DopplerAnalyzer:
    def __init__(self):
        self.hist = deque()
        self.spectrum_freqs = None
        self.spectrum_power = None
        self.band_energy = {name: 0.0 for name, _, _ in DOPPLER_BANDS}
        self.last_analysis_ts = 0.0
        self.wf_history_sec = 300.0
        self.wf_history = deque()
        self.baseline_wf = 500.0
        self.baseline_ready = False
        self.breath_baseline = 20.0

        # ---- Vitals long-window analysis (breath + experimental HR) ----
        # Uses aggregate CSI amplitude (mean across subcarriers) over VITALS_WINDOW_SEC,
        # giving much finer frequency resolution than the 2s motion FFT.
        self.vitals_hist = deque()          # (ts, aggregate_amplitude)
        self.last_vitals_ts = 0.0
        self.vitals_breath_bpm = None       # smoothed (EMA), None = insufficient signal
        self.vitals_heart_bpm = None
        self._vitals_breath_raw = None      # unsmoothed intermediate
        self._vitals_heart_raw = None

    def _trim(self, now_ts):
        cutoff = now_ts - DOPPLER_WINDOW_SEC * 1.2
        while self.hist and self.hist[0][0] < cutoff:
            self.hist.popleft()
        # Also trim vitals buffer
        vit_cutoff = now_ts - VITALS_WINDOW_SEC * 1.2
        while self.vitals_hist and self.vitals_hist[0][0] < vit_cutoff:
            self.vitals_hist.popleft()

    def push(self, csi_row):
        self.hist.append((csi_row["ts"], csi_row["amplitude"]))
        # Aggregate amplitude for vitals — one scalar per packet
        agg = float(np.mean(csi_row["amplitude"]))
        self.vitals_hist.append((csi_row["ts"], agg))
        self._trim(csi_row["ts"])

    def maybe_analyze(self, now_ts):
        if now_ts - self.last_analysis_ts < DOPPLER_HOP_SEC:
            return False
        if len(self.hist) < int(CSI_EXPECTED_RATE_HZ * DOPPLER_WINDOW_SEC * 0.5):
            return False
        times = np.array([t for (t, _) in self.hist], dtype=np.float64)
        n_sub = min(a.shape[0] for a in [x[1] for x in self.hist])
        if n_sub < 4:
            return False
        amps = np.stack([a[:n_sub] for (_, a) in self.hist])
        t0, t1 = times[0], times[-1]
        n_samples = max(16, int((t1 - t0) * CSI_EXPECTED_RATE_HZ))
        if n_samples < 16:
            return False
        t_uniform = np.linspace(t0, t1, n_samples)
        uniform = np.empty((n_samples, n_sub), dtype=np.float32)
        for s in range(n_sub):
            uniform[:, s] = np.interp(t_uniform, times, amps[:, s])
        uniform -= uniform.mean(axis=0, keepdims=True)
        window = np.hanning(n_samples).astype(np.float32)[:, None]
        uniform *= window
        spec = np.abs(np.fft.rfft(uniform, axis=0))
        agg = spec.mean(axis=1)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / CSI_EXPECTED_RATE_HZ)
        for name, lo, hi in DOPPLER_BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            self.band_energy[name] = float(agg[mask].sum()) if mask.any() else 0.0
        self.spectrum_freqs = freqs
        self.spectrum_power = agg
        self.last_analysis_ts = now_ts
        wf_now = self.band_energy["walk"] + self.band_energy["fall"]
        self.wf_history.append((now_ts, wf_now))
        cutoff = now_ts - self.wf_history_sec
        while self.wf_history and self.wf_history[0][0] < cutoff:
            self.wf_history.popleft()
        if len(self.wf_history) > 100:
            vals = sorted(v for _, v in self.wf_history)
            self.baseline_wf = float(vals[len(vals) // 10])
            self.baseline_ready = True
        return True

    def maybe_analyze_vitals(self, now_ts):
        """Long-window FFT on aggregate amplitude → breath + heart rate.

        Returns True if the vitals analysis ran this call.
        """
        if now_ts - self.last_vitals_ts < VITALS_HOP_SEC:
            return False
        need_samples = int(VITALS_WINDOW_SEC * CSI_EXPECTED_RATE_HZ * 0.7)
        if len(self.vitals_hist) < need_samples:
            return False

        times = np.array([t for (t, _) in self.vitals_hist], dtype=np.float64)
        vals = np.array([v for (_, v) in self.vitals_hist], dtype=np.float32)
        t0, t1 = times[0], times[-1]
        span = t1 - t0
        if span < VITALS_WINDOW_SEC * 0.5:
            return False

        # Resample onto uniform grid at CSI rate for a clean FFT
        n = max(64, int(span * CSI_EXPECTED_RATE_HZ))
        t_u = np.linspace(t0, t1, n)
        v_u = np.interp(t_u, times, vals).astype(np.float32)
        v_u -= v_u.mean()                                  # DC removal
        v_u *= np.hanning(n).astype(np.float32)            # window
        spec = np.abs(np.fft.rfft(v_u))
        freqs = np.fft.rfftfreq(n, d=1.0 / CSI_EXPECTED_RATE_HZ)

        # Ignore the very-low-freq end (near-DC leakage bleeds into breath band)
        floor = float(spec[freqs > 0.05].mean()) if (freqs > 0.05).any() else float(spec.mean())

        def _peak(band_lo, band_hi, snr_threshold):
            mask = (freqs >= band_lo) & (freqs < band_hi)
            if not mask.any():
                return None
            idxs = np.where(mask)[0]
            band_powers = spec[idxs]
            if len(band_powers) < 2:
                return None
            local_peak_local_idx = int(np.argmax(band_powers))
            peak_val = float(band_powers[local_peak_local_idx])
            if floor <= 0 or peak_val < floor * snr_threshold:
                return None
            # Convert local idx back to global for parabolic interpolation
            global_idx = int(idxs[local_peak_local_idx])
            hz = _parabolic_peak_hz(spec, freqs, global_idx)
            return hz * 60.0  # → BPM

        # Breath (respiratory): 6-36 BPM
        br = _peak(0.10, 0.60, snr_threshold=3.0)
        self._vitals_breath_raw = br
        if br is not None:
            if self.vitals_breath_bpm is None:
                self.vitals_breath_bpm = br
            else:
                a = VITALS_SMOOTH_ALPHA
                self.vitals_breath_bpm = a * br + (1 - a) * self.vitals_breath_bpm
        # Do NOT clear existing smoothed value if signal briefly weakens — hold it.

        # Heart: 48-180 BPM (experimental — needs higher SNR bar)
        hr = _peak(0.80, 3.00, snr_threshold=4.5)
        self._vitals_heart_raw = hr
        if hr is not None:
            if self.vitals_heart_bpm is None:
                self.vitals_heart_bpm = hr
            else:
                a = VITALS_SMOOTH_ALPHA
                self.vitals_heart_bpm = a * hr + (1 - a) * self.vitals_heart_bpm

        self.last_vitals_ts = now_ts
        return True


class FallDetectorV2:
    BURST_MIN_R = 2.4
    BURST_PEAK_MAX_R = 4.4
    STILLNESS_MAX_R = 1.8
    STILLNESS_RANGE_R = 0.60
    STILLNESS_RANGE_MIN_R = 0.12
    STILLNESS_RESUME = 1.3
    STILLNESS_SEC = 6.0
    BURST_MAX_SEC = 6.0
    COOLDOWN_SEC = 15.0

    def __init__(self):
        self.state = "idle"
        self.burst_ts = None
        self.burst_peak = 0.0
        self.stillness_start = None
        self.stillness_samples = []
        self.last_alert = -1e9

    def update(self, ts, walk, fall, baseline=500.0):
        wf = walk + fall
        if ts - self.last_alert < self.COOLDOWN_SEC:
            return False
        BURST_MIN = self.BURST_MIN_R * baseline
        BURST_PEAK_MAX = self.BURST_PEAK_MAX_R * baseline
        STILLNESS_MAX = self.STILLNESS_MAX_R * baseline
        STILLNESS_RANGE = self.STILLNESS_RANGE_R * baseline
        STILLNESS_RANGE_MIN = self.STILLNESS_RANGE_MIN_R * baseline
        if self.state == "idle":
            if wf > BURST_MIN:
                self.state, self.burst_ts, self.burst_peak = "burst", ts, wf
        elif self.state == "burst":
            if wf > self.burst_peak:
                self.burst_peak = wf
            if wf < STILLNESS_MAX:
                self.state = "stillness"
                self.stillness_start = ts
                self.stillness_samples = [wf]
            elif ts - self.burst_ts > self.BURST_MAX_SEC:
                self.state, self.burst_peak = "idle", 0.0
        elif self.state == "stillness":
            self.stillness_samples.append(wf)
            if wf > STILLNESS_MAX * self.STILLNESS_RESUME:
                self.state, self.burst_peak = "idle", 0.0
                return False
            still_range = max(self.stillness_samples) - min(self.stillness_samples)
            if still_range > STILLNESS_RANGE:
                self.state, self.burst_peak = "idle", 0.0
                return False
            if ts - self.stillness_start >= self.STILLNESS_SEC:
                if self.burst_peak > BURST_PEAK_MAX:
                    self.state, self.burst_peak = "idle", 0.0
                    return False
                if still_range < STILLNESS_RANGE_MIN:
                    self.state, self.burst_peak = "idle", 0.0
                    return False
                self.last_alert = ts
                self.state, self.burst_peak = "idle", 0.0
                return True
        return False


# =============================================================================
# BACKENDS
# =============================================================================

def _tcp_open(ip, port, timeout=0.8):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return False
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port)) == 0
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def discover_device(cfg, avoid_ips=None, log=print):
    label = cfg["label"]
    avoid_ips = set(avoid_ips or ())
    try:
        ip = socket.gethostbyname(cfg["mdns"])
        if ip not in avoid_ips and _tcp_open(ip, API_PORT):
            log(f"[{label}] mDNS -> {ip}")
            return ip
    except socket.gaierror:
        pass
    fb = cfg.get("fallback_ip")
    if fb and fb not in avoid_ips and _tcp_open(fb, API_PORT):
        log(f"[{label}] fallback {fb}")
        return fb
    hits = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(_tcp_open, f"{LAN_PREFIX}{i}", API_PORT): i for i in range(1, 255)}
        for f in concurrent.futures.as_completed(futs):
            if f.result():
                hits.append(futs[f])
    cands = [f"{LAN_PREFIX}{i}" for i in sorted(hits) if f"{LAN_PREFIX}{i}" not in avoid_ips]
    if not cands:
        raise RuntimeError(f"[{label}] not on LAN")
    log(f"[{label}] LAN scan picked {cands[0]}")
    return cands[0]


class ESPHomeBackend:
    def __init__(self, cfg, out_q):
        self.cfg = cfg
        self.label = cfg["label"]
        self.q = out_q
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"esph-{self.label}")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        expected = self.cfg["mdns"].rsplit(".", 1)[0]
        avoid = set()
        while not self._stop.is_set():
            client = None
            try:
                loop = asyncio.get_event_loop()
                host = await loop.run_in_executor(
                    None,
                    lambda: discover_device(self.cfg, avoid_ips=avoid,
                                            log=lambda m: self.q.put(("discover", self.label, m))),
                )
                client = APIClient(host, API_PORT, password="")
                await client.connect(login=True)
                info = await client.device_info()
                if info.name != expected:
                    self.q.put(("discover", self.label, f"IDENTITY MISMATCH at {host}"))
                    avoid.add(host)
                    await client.disconnect()
                    await asyncio.sleep(1)
                    continue
                entities, _ = await client.list_entities_services()
                key_to_id = {e.key: e.object_id for e in entities}
                self.q.put(("connected", self.label,
                            {"host": host, "version": info.esphome_version, "name": info.name}))

                def cb(state):
                    oid = key_to_id.get(state.key, f"key={state.key}")
                    val = getattr(state, "state", state)
                    self.q.put(("state", self.label, (oid, val, time.time())))

                client.subscribe_states(cb)
                while not self._stop.is_set():
                    await asyncio.sleep(1)
                    self.q.put(("heartbeat", self.label, time.time()))
            except Exception as e:
                self.q.put(("disconnected", self.label, f"{type(e).__name__}: {e}"))
                if client is not None:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                # Short sleep with stop check
                for _ in range(50):
                    if self._stop.is_set():
                        return
                    await asyncio.sleep(0.1)


class CSIBackend:
    def __init__(self, port, baud, out_q):
        self.port = port
        self.baud = baud
        self.q = out_q
        self.analyzer = DopplerAnalyzer()
        self.fall_det = FallDetectorV2()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="csi-backend")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        buf_txt = ""
        ser = None
        last_emit = 0.0
        while not self._stop.is_set():
            try:
                ser = serial.Serial(self.port, self.baud, timeout=0.01)
                self.q.put(("connected", "CSI",
                            {"host": self.port, "name": f"CSI {self.port}", "version": "-"}))
                while not self._stop.is_set():
                    try:
                        raw = ser.read(ser.in_waiting or 1)
                    except Exception as e:
                        raise RuntimeError(f"serial read: {e}")
                    if raw:
                        buf_txt += raw.decode("utf-8", errors="ignore")
                        while "\n" in buf_txt:
                            line, buf_txt = buf_txt.split("\n", 1)
                            csi = parse_csi_line(line)
                            if csi is not None:
                                self.analyzer.push(csi)
                    now = time.time()
                    if self.analyzer.maybe_analyze(now):
                        walk = self.analyzer.band_energy["walk"]
                        fall = self.analyzer.band_energy["fall"]
                        wf = walk + fall
                        # Send a "sample" event on every analyze so plot stays smooth
                        self.q.put(("state", "CSI", ("sample", wf, now)))
                        if self.fall_det.update(now, walk, fall, baseline=self.analyzer.baseline_wf):
                            self.q.put(("state", "CSI", ("fall_event", True, now)))
                    # Vitals analysis (separate longer-window FFT for breath + HR)
                    self.analyzer.maybe_analyze_vitals(now)
                    if now - last_emit > 0.5:
                        last_emit = now
                        state = self._derive_state(now)
                        self.q.put(("state", "CSI", ("derived", state, now)))
                        self.q.put(("heartbeat", "CSI", now))
                    time.sleep(0.005)
            except Exception as e:
                msg = str(e)
                if "PermissionError" in msg or "Zugriff" in msg or "in use" in msg or "denied" in msg.lower():
                    msg = f"{self.port} busy — close legacy/kisum_radar_monitor.py first ({msg})"
                self.q.put(("disconnected", "CSI", msg))
                try:
                    if ser is not None:
                        ser.close()
                except Exception:
                    pass
                # sleep 3 sec but check stop flag every 100ms
                for _ in range(30):
                    if self._stop.is_set():
                        return
                    time.sleep(0.1)

    def _derive_state(self, now):
        be = self.analyzer.band_energy
        baseline = max(self.analyzer.baseline_wf, 50.0)
        wf = be["walk"] + be["fall"]
        present = wf > 1.5 * baseline
        # Motion classification (thresholds are RATIOS of walk+fall energy vs
        # the rolling 5-min p10 baseline — self-adapting per environment).
        if not present:
            motion = "STILL"      # sitting/lying, no meaningful motion
        elif wf < 2.5 * baseline:
            motion = "LIGHT"      # typing, reading, small limb movement
        elif wf < 5.0 * baseline:
            motion = "ACTIVE"     # normal walking, household activity
        else:
            motion = "INTENSE"    # running, exercising, big movements
        # Use dedicated long-window vitals analyzer results (better freq resolution
        # + parabolic interp + EMA smoothing). See DopplerAnalyzer.maybe_analyze_vitals.
        breath_bpm = self.analyzer.vitals_breath_bpm
        heart_bpm = self.analyzer.vitals_heart_bpm

        return {
            "present": present,
            "motion": motion,
            "breath_bpm": breath_bpm,
            "heart_bpm": heart_bpm,
            "bands": dict(be),
            "baseline_wf": baseline,
            "fall_threshold_wf": FallDetectorV2.BURST_MIN_R * baseline,
        }


class PoseBackend:
    """Runs webcam + MediaPipe pose + heatmap render in a background thread.

    Exposes the latest processed BGR frame via get_latest_frame() for the UI.
    """
    def __init__(self, webcam_index, mode, palette_name, smooth_strength, out_q):
        self.webcam_index = webcam_index
        self.mode = mode
        self.palette_cv = PALETTES.get(palette_name)
        self.smooth_strength = smooth_strength
        self.q = out_q
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._stop = threading.Event()
        self._cap = None  # exposed so _on_close can release camera from main thread
        self._cap_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True, name="pose-backend")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def force_release_camera(self):
        """Release the cv2 VideoCapture object IMMEDIATELY, bypassing the
        pose thread. Critical for clean shutdown — camera drivers (DirectShow
        and MSMF alike) will otherwise hold pending I/O that pins the process
        object in the Windows kernel and blocks parent-terminal prompt return.
        """
        with self._cap_lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def get_latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _run(self):
        try:
            pose_path = ensure_model("pose_landmarker_lite.task")
        except Exception as e:
            self.q.put(("disconnected", "POSE", f"model download failed: {e}"))
            return

        def _make_landmarker():
            return create_pose_landmarker(pose_path, want_seg=(self.mode == "heatmap"))

        try:
            landmarker = _make_landmarker()
        except Exception as e:
            self.q.put(("disconnected", "POSE", f"pose init failed: {e}"))
            return

        smoother = LandmarkSmoother(strength=self.smooth_strength)
        # Tracker recovery: MediaPipe VIDEO-mode tracker can get stuck after
        # rapid motion (falls, camera bumps) and refuse to re-detect the
        # subject. If we go too long without any landmarks while camera IS
        # streaming, we recreate the landmarker from scratch to force fresh
        # detection.
        POSE_LOST_RESET_SEC = 8.0
        last_pose_ts = time.time()

        while not self._stop.is_set():
            cap = None
            try:
                # MSMF (Media Foundation) — releases cleanly on shutdown, but
                # startup takes 10-15 sec on some USB webcams. DirectShow
                # opens fast (<1s) but leaves pending I/O that pins the process
                # in Windows kernel and blocks the parent terminal prompt from
                # returning (verified 2026-07-30). We prefer MSMF; user waits.
                self.q.put(("discover", "POSE",
                            "opening webcam via MSMF (~10-15s, please wait)…"))
                t_open = time.time()
                cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_MSMF)
                if not cap.isOpened():
                    self.q.put(("discover", "POSE", "MSMF failed, trying DirectShow"))
                    cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.webcam_index)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open webcam {self.webcam_index}")
                open_secs = time.time() - t_open
                self.q.put(("discover", "POSE", f"webcam opened in {open_secs:.1f}s"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Expose cap for force_release_camera()
                with self._cap_lock:
                    self._cap = cap
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.q.put(("connected", "POSE",
                            {"host": f"webcam:{self.webcam_index}",
                             "name": f"MediaPipe {self.mode}", "version": f"{w}x{h}"}))

                t_start = time.time()
                last_landmarks = None
                last_pose_seg = None
                while not self._stop.is_set():
                    # Check if main thread released the camera under us
                    with self._cap_lock:
                        if self._cap is None:
                            break
                    ok, frame_bgr = cap.read()
                    if not ok:
                        # Could be a benign transient — but if we're stopping, exit
                        if self._stop.is_set():
                            break
                        raise RuntimeError("webcam read failed")
                    ts_ms = int((time.time() - t_start) * 1000)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result = landmarker.detect_for_video(mp_image, ts_ms)
                    if result.pose_landmarks and len(result.pose_landmarks) > 0:
                        raw_lm = result.pose_landmarks[0]
                        last_landmarks = smoother.smooth(raw_lm, ts_ms / 1000.0)
                        last_pose_ts = time.time()
                    else:
                        last_landmarks = None
                        # Auto-recover from tracker stuck-state after rapid motion
                        if time.time() - last_pose_ts > POSE_LOST_RESET_SEC:
                            self.q.put(("discover", "POSE",
                                        f"pose lost {POSE_LOST_RESET_SEC:.0f}s — resetting tracker"))
                            try:
                                landmarker.close()
                            except Exception:
                                pass
                            try:
                                landmarker = _make_landmarker()
                            except Exception as e:
                                self.q.put(("disconnected", "POSE",
                                            f"landmarker reset failed: {e}"))
                                raise
                            # Reset smoother too — old filter state is stale
                            smoother = LandmarkSmoother(strength=self.smooth_strength)
                            last_pose_ts = time.time()  # avoid reset loop
                    if hasattr(result, "segmentation_masks") and result.segmentation_masks:
                        last_pose_seg = result.segmentation_masks[0].numpy_view().copy()
                    else:
                        last_pose_seg = None

                    frame_out = apply_anonymization(frame_bgr, None, last_landmarks,
                                                    self.mode, COLOUR_SILHOUETTE_DEFAULT,
                                                    self.palette_cv, pose_seg=last_pose_seg)
                    if last_landmarks is not None:
                        draw_skeleton(frame_out, last_landmarks,
                                       frame_bgr.shape[1], frame_bgr.shape[0])
                    with self._frame_lock:
                        self._latest_frame = frame_out
                    self.q.put(("heartbeat", "POSE", time.time()))
            except Exception as e:
                self.q.put(("disconnected", "POSE", f"{type(e).__name__}: {e}"))
                with self._cap_lock:
                    self._cap = None
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                # sleep with stop check
                for _ in range(30):
                    if self._stop.is_set():
                        return
                    time.sleep(0.1)
        # loop exited (stop set) — release webcam cleanly
        with self._cap_lock:
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


class PhoneMockup(tk.Frame):
    """Simulated phone screen showing latest Telegram alerts as notification cards.

    Sits below the pose heatmap panel and serves as a Family App preview: what
    the family caregiver sees on their phone when the system fires an alert.
    """
    BEZEL_COLOUR   = "#111114"
    BEZEL_BORDER   = "#3a3a3f"
    SCREEN_COLOUR  = "#0d0d10"
    APP_BAR        = "#1e5cba"
    CARD_COLOUR    = "#1c1c1f"
    CARD_BORDER    = "#2a2a2f"
    CARD_TEXT      = "#ececec"
    CARD_MUTED     = "#8a8a90"
    EMPTY_TEXT     = "#4a4a50"
    N_CARDS = 3

    def __init__(self, parent, width=340, height=430):
        super().__init__(parent, bg=BG)
        self.w, self.h = width, height

        # Outer bezel — heavy border simulates phone edge
        bezel = tk.Frame(self, bg=self.BEZEL_COLOUR, width=width, height=height,
                         highlightthickness=3, highlightbackground=self.BEZEL_BORDER)
        bezel.pack()
        bezel.pack_propagate(False)

        # App bar (no fake status bar above — cleaner, avoids stretched glyphs)
        appbar = tk.Frame(bezel, bg=self.APP_BAR, height=56)
        appbar.pack(fill="x", padx=6, pady=(6, 0))
        appbar.pack_propagate(False)
        tk.Label(appbar, text="Family App  ·  Ambient Care",
                 fg="white", bg=self.APP_BAR,
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=16)
        tk.Label(appbar, text="🔔", fg="white", bg=self.APP_BAR,
                 font=("Segoe UI Emoji", 14)).pack(side="right", padx=14, pady=14)

        # Content area
        content = tk.Frame(bezel, bg=self.SCREEN_COLOUR)
        content.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Header row: section title + card count hint
        header_row = tk.Frame(content, bg=self.SCREEN_COLOUR)
        header_row.pack(fill="x", padx=6, pady=(10, 4))
        tk.Label(header_row, text="Latest alerts",
                 fg="#b4b4bc", bg=self.SCREEN_COLOUR,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=10)
        tk.Label(header_row, text=f"top {self.N_CARDS}",
                 fg=self.EMPTY_TEXT, bg=self.SCREEN_COLOUR,
                 font=("Segoe UI", 8)).pack(side="right", padx=10)

        # Cards wrap (holds either empty label OR N cards)
        self.cards_wrap = tk.Frame(content, bg=self.SCREEN_COLOUR)
        self.cards_wrap.pack(fill="both", expand=True, padx=8, pady=(2, 6))

        # Empty state — shown when no alerts (packed dynamically by update_alerts)
        self.empty_lbl = tk.Label(
            self.cards_wrap,
            text="🔕\n\nNo alerts yet\nThe system is monitoring quietly.",
            fg=self.EMPTY_TEXT, bg=self.SCREEN_COLOUR,
            font=("Segoe UI Emoji", 11), justify="center",
        )

        # Card widgets (packed on demand by update_alerts)
        self.cards = []
        for i in range(self.N_CARDS):
            card = tk.Frame(self.cards_wrap, bg=self.CARD_COLOUR, height=82,
                            highlightthickness=1, highlightbackground=self.CARD_BORDER)
            card.pack_propagate(False)
            side_bar = tk.Frame(card, bg=self.CARD_BORDER, width=5)
            side_bar.pack(side="left", fill="y")
            body = tk.Frame(card, bg=self.CARD_COLOUR)
            body.pack(side="left", fill="both", expand=True, padx=(10, 10), pady=8)
            top_row = tk.Frame(body, bg=self.CARD_COLOUR)
            top_row.pack(fill="x")
            title_lbl = tk.Label(top_row, text="", fg=self.CARD_TEXT,
                                 bg=self.CARD_COLOUR,
                                 font=("Segoe UI Emoji", 11, "bold"), anchor="w")
            title_lbl.pack(side="left")
            ts_lbl = tk.Label(top_row, text="", fg=self.CARD_MUTED,
                              bg=self.CARD_COLOUR,
                              font=("Segoe UI", 8), anchor="e")
            ts_lbl.pack(side="right")
            text_lbl = tk.Label(body, text="", fg=self.CARD_TEXT,
                                bg=self.CARD_COLOUR,
                                font=("Segoe UI", 9), anchor="w", justify="left",
                                wraplength=width - 70)
            text_lbl.pack(fill="x", pady=(4, 0))
            self.cards.append({
                "card": card, "side_bar": side_bar, "body": body,
                "top_row": top_row, "title": title_lbl,
                "ts": ts_lbl, "text": text_lbl,
            })

    def _set_card_bg(self, cd, bg, border):
        cd["card"].config(bg=bg, highlightbackground=border)
        cd["body"].config(bg=bg)
        cd["top_row"].config(bg=bg)
        cd["title"].config(bg=bg)
        cd["ts"].config(bg=bg)
        cd["text"].config(bg=bg)

    def update_alerts(self, alerts, now):
        if not alerts:
            for cd in self.cards:
                cd["card"].pack_forget()
            self.empty_lbl.pack(fill="both", expand=True, pady=24)
            return
        self.empty_lbl.pack_forget()
        for i, cd in enumerate(self.cards):
            if i >= len(alerts):
                cd["card"].pack_forget()
                continue
            ts, cat, text, colour = alerts[i]
            cd["card"].pack(fill="x", padx=4, pady=4)
            cd["title"].config(text=cat, fg=colour)
            cd["ts"].config(text=time.strftime("%H:%M:%S", time.localtime(ts)))
            cd["text"].config(text=text, fg=self.CARD_TEXT)
            cd["side_bar"].config(bg=colour)
            if i == 0 and (now - ts) < 5.0:
                if colour == RED:
                    flash_bg = "#3a1717"
                elif colour == GREEN:
                    flash_bg = "#1e3020"
                else:
                    flash_bg = "#2a2419"
                self._set_card_bg(cd, flash_bg, colour)
            else:
                self._set_card_bg(cd, self.CARD_COLOUR, self.CARD_BORDER)


class VideoPanel(tk.Frame):
    """Tk widget that displays the latest frame from PoseBackend, ~20 fps."""
    def __init__(self, parent, backend, size=POSE_DISPLAY_SIZE):
        super().__init__(parent, bg=BG)
        self.backend = backend
        self.w, self.h = size
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x")
        self.title_lbl = tk.Label(header, text="Pose Heatmap (webcam)  ", fg=FG, bg=BG,
                                  font=("Segoe UI", 9, "bold"))
        self.title_lbl.pack(side="left")
        self.info_lbl = tk.Label(header, text="", fg=DIM, bg=BG, font=("Consolas", 9))
        self.info_lbl.pack(side="left")
        # Fixed-size dark canvas placeholder — image and text are SEPARATE widgets
        # to avoid Tk pyimage GC race when config()-ing text on a label whose
        # image was replaced.
        placeholder = Image.new("RGB", (self.w, self.h), (14, 17, 22))
        self._photo = ImageTk.PhotoImage(placeholder)
        self.image_lbl = tk.Label(self, image=self._photo, bg="#0e1116",
                                  bd=1, highlightthickness=1, highlightbackground=BORDER)
        self.image_lbl.pack()
        # Loading overlay — positioned on top of the image, destroyed on first frame
        self.loading_lbl = tk.Label(self.image_lbl,
                                    text="opening webcam (MSMF, ~10-15s)…",
                                    fg=DIM, bg="#0e1116", font=("Segoe UI", 12))
        self.loading_lbl.place(relx=0.5, rely=0.5, anchor="center")
        self._first_frame_shown = False
        self._last_update_ts = 0.0
        self._frame_count = 0
        self.after(50, self._tick)

    def _tick(self):
        try:
            frame = self.backend.get_latest_frame()
            if frame is not None:
                resized = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                new_photo = ImageTk.PhotoImage(img)
                self.image_lbl.config(image=new_photo)
                self._photo = new_photo   # keep reference AFTER config to avoid GC race
                if not self._first_frame_shown:
                    # Destroy the loading overlay once we have a real frame
                    try:
                        self.loading_lbl.destroy()
                    except Exception:
                        pass
                    self._first_frame_shown = True
                self._frame_count += 1
                now = time.time()
                if now - self._last_update_ts > 1.0:
                    fps = self._frame_count / max(1e-3, (now - self._last_update_ts))
                    self.info_lbl.config(text=f"{fps:.0f} fps  ({POSE_MODE}/{POSE_PALETTE})")
                    self._last_update_ts = now
                    self._frame_count = 0
        except Exception as e:
            # Never let a rendering error kill the tick loop
            try:
                self.info_lbl.config(text=f"tick error: {type(e).__name__}")
            except Exception:
                pass
        # Always reschedule, even on error
        self.after(50, self._tick)


def find_windows_by_title_substring(needle: str):
    """Return list of (hwnd, title, x, y, w, h) for visible top-level windows
    whose title contains `needle` (case-insensitive). Windows-only.
    """
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    results = []
    needle_low = needle.lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum_proc(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if needle_low in title.lower():
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > 100 and h > 100:  # ignore tooltips/tiny windows
                    results.append((int(hwnd), title, rect.left, rect.top, w, h))
        return True

    user32.EnumWindows(_enum_proc, 0)
    return results


class TelegramNotifier:
    """Fire-and-forget notification client using Telegram Bot API.

    Send: background thread, rate-limited per category.
    Receive callbacks (inline button presses): long-polling background thread.
    All results posted to shared queue for main-thread UI update.
    """
    def __init__(self, bot_token: str, chat_id: str, out_q=None):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.q = out_q  # shared App queue (for callback events)
        self._last_send_ts = {}   # category → last-send timestamp
        self._update_offset = 0   # for getUpdates de-dup
        self._stop_polling = threading.Event()

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def start_polling(self):
        """Start background long-polling for callback_query events."""
        if not self.is_configured() or self.q is None:
            return
        threading.Thread(target=self._poll_loop, daemon=True,
                         name="telegram-poll").start()

    def stop_polling(self):
        self._stop_polling.set()

    def send(self, text: str, category: str = "default", cooldown_sec: float = 5.0,
             inline_keyboard=None):
        """Queue a message. inline_keyboard = list of button rows, each button:
        {'text': label, 'callback_data': tag}."""
        if not self.is_configured():
            return False
        now = time.time()
        last = self._last_send_ts.get(category, 0)
        if now - last < cooldown_sec:
            return False
        self._last_send_ts[category] = now
        threading.Thread(target=self._do_send, args=(text, inline_keyboard),
                         daemon=True).start()
        return True

    def _do_send(self, text: str, inline_keyboard=None):
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            if inline_keyboard:
                payload["reply_markup"] = json.dumps(
                    {"inline_keyboard": inline_keyboard}
                )
            data = urllib.parse.urlencode(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            urllib.request.urlopen(req, timeout=6).read()
        except Exception:
            pass  # swallow — telegram outage/no-net must never crash the monitor

    def _poll_loop(self):
        """Long-poll getUpdates for callback_query events (button presses)."""
        while not self._stop_polling.is_set():
            try:
                params = urllib.parse.urlencode({
                    "offset": self._update_offset,
                    "timeout": 25,
                    "allowed_updates": json.dumps(["callback_query"]),
                })
                url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates?{params}"
                r = urllib.request.urlopen(url, timeout=30)
                body = json.loads(r.read().decode("utf-8"))
                if not body.get("ok"):
                    time.sleep(2)
                    continue
                for update in body.get("result", []):
                    self._update_offset = update["update_id"] + 1
                    if "callback_query" in update:
                        cb = update["callback_query"]
                        # Answer callback to stop the button's loading spinner
                        self._answer_callback(cb["id"], "Received")
                        # Post to App queue for UI handling
                        try:
                            self.q.put(("telegram_callback", "SYS", cb))
                        except Exception:
                            pass
            except Exception:
                # Any error (network hiccup, telegram outage) — wait then retry
                time.sleep(3)

    def _answer_callback(self, callback_id: str, text: str = ""):
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
            data = urllib.parse.urlencode({
                "callback_query_id": callback_id,
                "text": text,
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass


class ScreenRecorder:
    """Records the app window (screen region) into an MP4 file. Toggle via start/stop."""
    def __init__(self, root_widget, out_dir):
        self.root = root_widget
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.thread = None
        self.writer = None
        self.recording = False
        self.file_path = None
        self.frames_written = 0
        self.region = None
        self._start_ts = 0.0

    def start(self, extra_region=None):
        """Start recording. If extra_region=(x, y, w, h) is given, capture the
        UNION of the monitor window rect AND that region — so one MP4 contains
        both the monitor and (e.g.) the Telegram Web Chrome window side-by-side.
        """
        if self.recording:
            return None
        self.root.update()
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if extra_region:
            ex, ey, ew, eh = extra_region
            x2 = min(x, ex)
            y2 = min(y, ey)
            xr = max(x + w, ex + ew)
            yr = max(y + h, ey + eh)
            x, y, w, h = x2, y2, xr - x2, yr - y2
        # Even dims required by some codecs
        w -= w % 2
        h -= h % 2
        if w <= 0 or h <= 0:
            return None
        self.region = {"left": x, "top": y, "width": w, "height": h}
        self.file_path = os.path.join(
            self.out_dir, f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.file_path, fourcc, float(SCREEN_REC_FPS), (w, h))
        if not self.writer.isOpened():
            self.writer = None
            return None
        self.frames_written = 0
        self.recording = True
        self._start_ts = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True, name="screen-rec")
        self.thread.start()
        return self.file_path

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        t = self.thread
        if t is not None:
            t.join(timeout=2.0)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.file_path

    def _run(self):
        interval = 1.0 / float(SCREEN_REC_FPS)
        next_t = time.time()
        # Each thread must create its own mss instance
        with mss.mss() as sct:
            while self.recording:
                now = time.time()
                if now < next_t:
                    time.sleep(max(0.0, next_t - now))
                try:
                    shot = sct.grab(self.region)
                    arr = np.array(shot)
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                    if self.writer is not None:
                        self.writer.write(arr)
                        self.frames_written += 1
                except Exception:
                    break
                next_t += interval

    def status(self):
        if not self.recording:
            return None
        return {
            "elapsed": time.time() - self._start_ts,
            "frames": self.frames_written,
            "path": self.file_path,
        }


# =============================================================================
# FUSION
# =============================================================================

def fuse(fda2_st, bha2_st, csi_st):
    out = {}
    # Presence
    f_p = fda2_st.get("person_information")
    b_p = bha2_st.get("person_information")
    c_p = csi_st.get("present") if csi_st else None
    if any(x is True for x in (f_p, b_p, c_p)):
        out["presence"] = "PRESENT"
    elif any(x is not None for x in (f_p, b_p, c_p)) and all(x is False for x in (f_p, b_p, c_p) if x is not None):
        out["presence"] = "NO PERSON"
    else:
        out["presence"] = "WAITING"

    # Motion
    c_m = csi_st.get("motion") if csi_st else None
    out["motion"] = c_m or "—"
    out["motion_source"] = "CSI"

    OF_NAME = DISPLAY_NAMES["BHA2"]   # "mmWave Living Room"
    BR_NAME = DISPLAY_NAMES["FDA2"]   # "mmWave Bedroom"

    # Breath — never surface an out-of-range value on the Decision column;
    # better to show "—" than an alarming false reading like "1/min".
    b_br = bha2_st.get("real-time_respiratory_rate")
    csi_br = csi_st.get("breath_bpm") if csi_st else None
    b_br_ok = b_br is not None and BHA2_BREATH_MIN <= b_br <= BHA2_BREATH_MAX
    csi_br_ok = csi_br is not None and BHA2_BREATH_MIN <= csi_br <= BHA2_BREATH_MAX
    if b_br_ok:
        out["breath"] = f"{b_br:.0f}/min"
        out["breath_source"] = OF_NAME
        out["breath_colour"] = "green"
    elif csi_br_ok:
        out["breath"] = f"~{csi_br:.0f}/min"
        out["breath_source"] = f"CSI ({OF_NAME} out of range)"
        out["breath_colour"] = "amber"
    else:
        # Both sensors unreliable — hide the raw value, show placeholder
        out["breath"] = "—"
        out["breath_source"] = "insufficient signal"
        out["breath_colour"] = "dim"

    # Heart — plain average of mmWave + CSI when both valid; if one is missing,
    # use the other. There is deliberately NO sanity check on divergence between
    # the two sources; the source label shows which path produced the value.
    b_hr = bha2_st.get("real-time_heart_rate")
    csi_hr = csi_st.get("heart_bpm") if csi_st else None
    b_ok = b_hr is not None and BHA2_HEART_MIN <= b_hr <= BHA2_HEART_MAX
    c_ok = csi_hr is not None and BHA2_HEART_MIN <= csi_hr <= BHA2_HEART_MAX
    if b_ok and c_ok:
        avg = (b_hr + csi_hr) / 2.0
        out["heart"] = f"{avg:.0f} bpm"
        out["heart_source"] = "mmWave+CSI"
        out["heart_colour"] = "green"
    elif b_ok:
        out["heart"] = f"{b_hr:.0f} bpm"
        out["heart_source"] = OF_NAME
        out["heart_colour"] = "green"
    elif c_ok:
        out["heart"] = f"~{csi_hr:.0f} bpm"
        out["heart_source"] = "CSI (experimental)"
        out["heart_colour"] = "amber"
    else:
        out["heart"] = "—"
        out["heart_source"] = "insufficient signal"
        out["heart_colour"] = "dim"

    # Fall
    f_f = fda2_st.get("falling_information")
    if f_f is True:
        out["fall"] = "FALL"
        out["fall_colour"] = "red"
    elif f_f is False:
        out["fall"] = "ok"
        out["fall_colour"] = "green"
    else:
        out["fall"] = "—"
        out["fall_colour"] = "dim"
    out["fall_source"] = BR_NAME

    return out


# =============================================================================
# UI
# =============================================================================

BG = "#0e1116"
PANEL = "#161b22"
PANEL_FUSE = "#1a2028"  # slightly brighter for center column
BORDER = "#30363d"
BORDER_FUSE = "#4b5768"
FG = "#d4d4d4"
DIM = "#5c6370"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#d2a8ff"

COLOUR_MAP = {"green": GREEN, "red": RED, "amber": AMBER, "dim": DIM, "fg": FG, "blue": BLUE}


class ThreeColRow:
    """One horizontal row spanning 3 columns: (left cell, center fused cell, right cell).

    Owner attaches each cell frame to the appropriate column parent.
    """

    def __init__(self, left_parent, center_parent, right_parent, title,
                 left_labels, right_labels, center_font_size=18):
        # LEFT cell
        self.left_frame = tk.Frame(left_parent, bg=PANEL, bd=0,
                                   highlightthickness=1, highlightbackground=BORDER)
        tk.Label(self.left_frame, text=title, fg=DIM, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        self.left_labels = []
        for src, initial in left_labels:
            row = tk.Frame(self.left_frame, bg=PANEL)
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=src, fg=DIM, bg=PANEL, font=("Consolas", 10),
                     width=6, anchor="w").pack(side="left")
            v = tk.Label(row, text=initial, fg=FG, bg=PANEL, font=("Consolas", 11))
            v.pack(side="left")
            self.left_labels.append(v)
        tk.Frame(self.left_frame, height=6, bg=PANEL).pack()

        # CENTER cell (bigger, brighter background, larger text)
        self.center_frame = tk.Frame(center_parent, bg=PANEL_FUSE, bd=0,
                                     highlightthickness=1, highlightbackground=BORDER_FUSE)
        tk.Label(self.center_frame, text=title, fg=DIM, bg=PANEL_FUSE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        center_body = tk.Frame(self.center_frame, bg=PANEL_FUSE)
        center_body.pack(fill="x", padx=10, pady=(0, 4))
        self.arrow_lbl = tk.Label(center_body, text="►►", fg=GREEN, bg=PANEL_FUSE,
                                  font=("Segoe UI", 12, "bold"))
        self.arrow_lbl.pack(side="left", padx=(0, 6))
        self.center_val = tk.Label(center_body, text="—", fg=GREEN, bg=PANEL_FUSE,
                                   font=("Segoe UI", center_font_size, "bold"))
        self.center_val.pack(side="left")
        self.center_source = tk.Label(self.center_frame, text="", fg=DIM, bg=PANEL_FUSE,
                                      font=("Consolas", 9))
        self.center_source.pack(anchor="w", padx=10, pady=(0, 6))

        # RIGHT cell
        self.right_frame = tk.Frame(right_parent, bg=PANEL, bd=0,
                                    highlightthickness=1, highlightbackground=BORDER)
        tk.Label(self.right_frame, text=title, fg=DIM, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        self.right_labels = []
        for src, initial in right_labels:
            row = tk.Frame(self.right_frame, bg=PANEL)
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=src, fg=DIM, bg=PANEL, font=("Consolas", 10),
                     width=6, anchor="w").pack(side="left")
            v = tk.Label(row, text=initial, fg=FG, bg=PANEL, font=("Consolas", 11))
            v.pack(side="left")
            self.right_labels.append(v)
        tk.Frame(self.right_frame, height=6, bg=PANEL).pack()

    def pack(self):
        self.left_frame.pack(fill="x", pady=3, padx=2)
        self.center_frame.pack(fill="x", pady=3, padx=2)
        self.right_frame.pack(fill="x", pady=3, padx=2)

    def set_left(self, idx, text, colour="fg"):
        self.left_labels[idx].config(text=text, fg=COLOUR_MAP.get(colour, FG))

    def set_right(self, idx, text, colour="fg"):
        self.right_labels[idx].config(text=text, fg=COLOUR_MAP.get(colour, FG))

    def set_center(self, text, source, colour="green"):
        c = COLOUR_MAP.get(colour, GREEN)
        self.center_val.config(text=text, fg=c)
        self.arrow_lbl.config(fg=c)
        self.center_source.config(text=f"(source: {source})" if source else "")


class LinePlot(tk.Frame):
    """Canvas-based real-time line chart with two horizontal reference lines."""

    def __init__(self, parent, title, height=140,
                 window_sec=PLOT_WINDOW_SEC,
                 low_ref_label="baseline", low_ref_colour="#37484d",
                 high_ref_label="fall threshold", high_ref_colour="#5c2a2a",
                 high_ref_text_colour="#a8615e",
                 value_prefix="latest", value_suffix="",
                 y_min_forced=None, y_max_forced=None,
                 line_colour=None):
        super().__init__(parent, bg=BG)
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x")
        self.title_lbl = tk.Label(header, text=title, fg=FG, bg=BG,
                                  font=("Segoe UI", 9, "bold"))
        self.title_lbl.pack(side="left", padx=(2, 0))
        self.info_lbl = tk.Label(header, text="", fg=DIM, bg=BG, font=("Consolas", 9))
        self.info_lbl.pack(side="right")
        self.canvas = tk.Canvas(self, height=height, bg="#0a0d12",
                                highlightthickness=1, highlightbackground=BORDER)
        self.canvas.pack(fill="x", pady=(2, 0))
        self.data = deque(maxlen=1500)  # (ts, value)
        self.low_ref = None
        self.high_ref = None
        # Config
        self.window_sec = window_sec
        self.low_ref_label = low_ref_label
        self.low_ref_colour = low_ref_colour
        self.high_ref_label = high_ref_label
        self.high_ref_colour = high_ref_colour
        self.high_ref_text_colour = high_ref_text_colour
        self.value_prefix = value_prefix
        self.value_suffix = value_suffix
        self.y_min_forced = y_min_forced
        self.y_max_forced = y_max_forced
        self.line_colour = line_colour or GREEN

    def add_sample(self, ts, value, low_ref=None, high_ref=None,
                   baseline=None, fall_threshold=None):
        # Backward compat: baseline/fall_threshold aliases
        if low_ref is None and baseline is not None:
            low_ref = baseline
        if high_ref is None and fall_threshold is not None:
            high_ref = fall_threshold
        self.data.append((ts, value))
        if low_ref is not None:
            self.low_ref = low_ref
        if high_ref is not None:
            self.high_ref = high_ref

    def redraw(self, now_ts):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 2 or h <= 2:
            return
        cutoff = now_ts - self.window_sec
        while self.data and self.data[0][0] < cutoff:
            self.data.popleft()
        if len(self.data) < 2:
            self.info_lbl.config(text="waiting for data…")
            return

        vals = [v for _, v in self.data]
        if self.y_max_forced is not None:
            vmax = self.y_max_forced
        else:
            vmax = max(vals + [self.high_ref or 0, (self.low_ref or 0) * 3, 100]) * 1.1
        if self.y_min_forced is not None:
            vmin = self.y_min_forced
        else:
            vmin = 0

        def x_of(t):
            return (t - cutoff) / self.window_sec * w

        def y_of(v):
            v = max(vmin, min(vmax, v))
            return h - (v - vmin) / max(vmax - vmin, 1e-9) * (h - 4) - 2

        if self.low_ref is not None:
            y = y_of(self.low_ref)
            self.canvas.create_line(0, y, w, y, fill=self.low_ref_colour, width=1, dash=(2, 3))
            self.canvas.create_text(w - 6, y - 8,
                                    text=f"{self.low_ref_label} {self.low_ref:.0f}",
                                    fill=DIM, anchor="e", font=("Consolas", 8))
        if self.high_ref is not None:
            y = y_of(self.high_ref)
            self.canvas.create_line(0, y, w, y, fill=self.high_ref_colour, width=1, dash=(4, 3))
            self.canvas.create_text(w - 6, y - 8,
                                    text=f"{self.high_ref_label} {self.high_ref:.0f}",
                                    fill=self.high_ref_text_colour, anchor="e", font=("Consolas", 8))

        pts = []
        for t, v in self.data:
            pts.extend([x_of(t), y_of(v)])
        self.canvas.create_line(*pts, fill=self.line_colour, width=1, smooth=False)

        last_t, last_v = self.data[-1]
        lx, ly = x_of(last_t), y_of(last_v)
        self.canvas.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill=self.line_colour, outline="")
        # Live sample rate — helps diagnose CSI dropouts vs display bugs
        rate = len(self.data) / self.window_sec
        rate_col = DIM if rate > 3.0 else AMBER if rate > 0.5 else RED
        self.info_lbl.config(
            text=(f"{self.value_prefix} = {last_v:.0f}{self.value_suffix}   "
                  f"({len(self.data)} samples, {rate:.1f}/s)"),
            fg=rate_col,
        )


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG)
        self.root.geometry("1600x1000")
        self.root.minsize(1300, 800)

        self.q = queue.Queue()

        self.state = {"FDA2": {}, "BHA2": {}, "CSI": {}}
        self.csi_derived = {}
        self.last_hb = {"FDA2": 0.0, "BHA2": 0.0, "CSI": 0.0, "POSE": 0.0}
        self.connected = {"FDA2": False, "BHA2": False, "CSI": False, "POSE": False}
        self.last_fall_csi_ts = None

        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_enabled = tk.BooleanVar(value=False)
        self.log_file = None
        self.log_writer = None
        self.log_last_write = 0.0

        # Pose backend (must be created BEFORE _build_ui because VideoPanel needs it)
        self.pose_backend = PoseBackend(WEBCAM_INDEX, POSE_MODE, POSE_PALETTE,
                                        POSE_SMOOTH, self.q)
        # Screen recorder (created here, wired to REC button in _build_ui)
        self.screen_rec = ScreenRecorder(self.root, DEMO_DIR)
        # Telegram notifier — pass self.q so callback events (button presses)
        # reach the App _pump loop and update UI.
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                         out_q=self.q)
        # Start polling for button responses from family
        self.telegram.start_polling()
        # Rect of the auto-launched Chrome Telegram-Web window (populated when
        # user clicks "Open Telegram"). Used by the screen recorder to record
        # a bounding box that includes both monitor and Telegram Web.
        self.telegram_web_rect = None
        # Inactivity tracking (person present but no motion)
        self._last_motion_ts = time.time()
        self._inactivity_alerted = False
        # Family-app-preview alert history (newest first)
        self.alerts_history = deque(maxlen=5)   # (ts, category, text, colour)
        self._latest_alert_ts = 0.0             # for flash animation timer

        self._build_ui()

        # Only start backends for sources that were configured; the others
        # show up red in the status row with a "not configured" event.
        self.backends = []
        for cfg in DEVICES:
            if cfg["mdns"]:
                self.backends.append(ESPHomeBackend(cfg, self.q))
            else:
                self.q.put(("disconnected", cfg["label"],
                            f"not configured (--{cfg['label'].lower()}-host)"))
        if CSI_PORT:
            self.backends.append(CSIBackend(CSI_PORT, CSI_BAUD, self.q))
        else:
            self.q.put(("disconnected", "CSI", "not configured (--csi-port)"))
        for b in self.backends:
            b.start()
        if not DISABLE_POSE:
            self.pose_backend.start()
        else:
            # Fake status entry so UI shows it disabled
            self.q.put(("disconnected", "POSE", "disabled via DISABLE_POSE flag"))

        # Clean shutdown when user closes the window (X button)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.after(50, self._pump)
        self.root.after(200, self._tick_ui)
        self.root.after(200, self._check_sigint)

        # Optional (--telegram-web): launch Chrome Telegram Web 1.5 sec after
        # startup — so the monitor window is fully positioned first, then
        # Chrome docks to its right.
        if AUTO_OPEN_TELEGRAM_WEB:
            self.root.after(1500, self._on_open_telegram_web)

    def _check_sigint(self):
        """Poll the SIGINT flag every 200ms so Ctrl+C in the terminal can also
        trigger a clean shutdown (Tk mainloop can't receive signals directly
        on Windows because it blocks in C code)."""
        if _SIGINT_RECEIVED[0]:
            self._on_close()
            return
        self.root.after(200, self._check_sigint)

    def _on_close_stop_telegram(self):
        try:
            self.telegram.stop_polling()
        except Exception:
            pass

    def _on_close(self):
        """Signal all backends to stop, then tear down the window.

        SHUTDOWN GUARANTEE — schedule a hard-kill BEFORE any cleanup, so even
        if cleanup or Tk teardown blocks, the process dies within 0.5 sec.
        cv2.VideoCapture (DirectShow), mss, and aioesphomeapi are all known to
        leave process handles pinned by the Windows kernel after normal exit,
        so we call TerminateProcess directly on ourselves as the last word.
        """
        # (1) Guarantee process death — spawn this FIRST, before anything else.
        # Strategy: launch a DETACHED PowerShell that kills us from OUTSIDE via
        # .NET Process.Kill (the only mechanism proven to reliably release the
        # Windows kernel process object — TerminateProcess-on-self and taskkill
        # both fail intermittently, leaving zombie handles that keep the parent
        # terminal blocked). External-kill is bulletproof.
        my_pid = os.getpid()
        try:
            DETACHED = 0x00000008        # subprocess.DETACHED_PROCESS
            NEW_PGRP = 0x00000200        # subprocess.CREATE_NEW_PROCESS_GROUP
            NO_WIN   = 0x08000000        # CREATE_NO_WINDOW
            ps_cmd = (
                f"Start-Sleep -Milliseconds 500; "
                f"try {{ [System.Diagnostics.Process]::GetProcessById({my_pid}).Kill() }} catch {{ }}; "
                f"Start-Sleep -Milliseconds 300; "
                f"try {{ Stop-Process -Id {my_pid} -Force -ErrorAction SilentlyContinue }} catch {{ }}"
            )
            subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                creationflags=DETACHED | NEW_PGRP | NO_WIN,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except Exception:
            pass

        # Also try in-process kill (fast path, may or may not work)
        def _self_kill():
            time.sleep(0.4)
            try:
                import ctypes
                h = ctypes.windll.kernel32.GetCurrentProcess()
                ctypes.windll.kernel32.TerminateProcess(h, 0)
            except Exception:
                pass
            try:
                os._exit(0)
            except Exception:
                pass
        threading.Thread(target=_self_kill, daemon=True).start()

        # (2) IMMEDIATELY release camera (from main thread, not the pose thread)
        # This is the critical step for clean shutdown — see PoseBackend
        # .force_release_camera docstring for why.
        try:
            self.pose_backend.force_release_camera()
        except Exception:
            pass

        # (3) Best-effort polite cleanup (may be interrupted by hard-exit)
        try:
            if self.screen_rec.recording:
                try:
                    self.screen_rec.stop()
                except Exception:
                    pass
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass
            for b in self.backends:
                try:
                    b.stop()
                except Exception:
                    pass
            try:
                self.pose_backend.stop()
            except Exception:
                pass
            self._on_close_stop_telegram()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---------------- UI ----------------

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(top, text=APP_TITLE, fg=FG, bg=BG,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        tk.Checkbutton(top, text="record CSV log", variable=self.log_enabled,
                       fg=FG, bg=BG, selectcolor=BG, activebackground=BG,
                       activeforeground=FG, font=("Segoe UI", 10),
                       command=self._on_toggle_log).pack(side="left", padx=(24, 0))
        self.log_status = tk.Label(top, text="", fg=DIM, bg=BG, font=("Consolas", 9))
        self.log_status.pack(side="left", padx=(6, 0))

        # REC button — records the whole window to MP4 (toggle)
        self.rec_btn = tk.Button(top, text="● REC", fg=RED, bg=PANEL,
                                 activebackground="#22272e", activeforeground=RED,
                                 font=("Segoe UI", 10, "bold"), bd=1, relief="raised",
                                 padx=8, pady=2, command=self._on_toggle_rec)
        self.rec_btn.pack(side="left", padx=(16, 0))
        self.rec_status = tk.Label(top, text="", fg=DIM, bg=BG, font=("Consolas", 9))
        self.rec_status.pack(side="left", padx=(6, 0))

        # (Telegram Web opens only with --telegram-web — see App.__init__
        # for the scheduled call. No manual button.)

        # (Telegram status indicator hidden — the Family App preview panel
        # already shows the alert delivery state visually.)

        self.clock_lbl = tk.Label(top, text="", fg=DIM, bg=BG, font=("Consolas", 10))
        self.clock_lbl.pack(side="right")

        # Status row
        src_row = tk.Frame(self.root, bg=BG)
        src_row.pack(fill="x", padx=12, pady=(2, 6))
        self.src_dots = {}
        self.src_info = {}
        for lbl in ("FDA2", "BHA2", "CSI", "POSE"):
            frame = tk.Frame(src_row, bg=BG)
            frame.pack(side="left", padx=(0, 20))
            dot = tk.Label(frame, text="●", fg=DIM, bg=BG, font=("Segoe UI", 14, "bold"))
            dot.pack(side="left")
            name = tk.Label(frame, text=DISPLAY_NAMES.get(lbl, lbl),
                            fg=FG, bg=BG, font=("Segoe UI", 10, "bold"))
            name.pack(side="left", padx=(4, 4))
            info = tk.Label(frame, text="…", fg=DIM, bg=BG, font=("Consolas", 9))
            info.pack(side="left")
            self.src_dots[lbl] = dot
            self.src_info[lbl] = info

        # === Main area: [left column: video + phone mockup] | [Notebook of zone tabs] ===
        main_row = tk.Frame(self.root, bg=BG)
        main_row.pack(fill="x", padx=12, pady=(4, 4))

        # Left column stacks: pose video on top, family-app phone mockup below.
        left_col_wrap = tk.Frame(main_row, bg=BG)
        left_col_wrap.pack(side="left", padx=(0, 8), anchor="n")

        # Live pose heatmap video (shared across all tabs — user moves the
        # physical webcam between zones as they switch tabs).
        self.video_panel = VideoPanel(left_col_wrap, self.pose_backend, size=POSE_DISPLAY_SIZE)
        self.video_panel.pack(anchor="n")

        # Phone mockup below video (Family App preview — where fall alerts land)
        self.phone_mockup = PhoneMockup(left_col_wrap,
                                        width=POSE_DISPLAY_SIZE[0], height=430)
        self.phone_mockup.pack(anchor="n", pady=(10, 0))

        # Right: notebook with two zone tabs (Living Room / Bedroom)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Zone.TNotebook", background=BG, borderwidth=0)
        # Use Segoe UI Emoji so emoji icons in tab labels render on Windows
        style.configure("Zone.TNotebook.Tab",
                        background=PANEL, foreground=DIM,
                        padding=(20, 8), font=("Segoe UI Emoji", 11, "bold"))
        style.map("Zone.TNotebook.Tab",
                  background=[("selected", PANEL_FUSE)],
                  foreground=[("selected", GREEN)])

        self.notebook = ttk.Notebook(main_row, style="Zone.TNotebook")
        self.notebook.pack(side="left", fill="both", expand=True)

        # -------- Living Room tab (desk zone: BHA2 vitals + CSI motion) --------
        self.office_frame = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.office_frame, text="  🛋  Living Room  ")

        # 3-column grid inside the office tab
        # Layout order (left → right): mmWave raw | CSI raw | Decision (fused)
        # Decision is on the right and visually elevated (via PANEL_FUSE
        # background inside ThreeColRow's center_frame).
        self.grid_container = tk.Frame(self.office_frame, bg=BG)
        self.grid_container.pack(fill="x", pady=(4, 0))
        self.left_col = tk.Frame(self.grid_container, bg=BG)      # mmWave raw
        self.right_col = tk.Frame(self.grid_container, bg=BG)     # CSI raw (middle visually)
        self.center_col = tk.Frame(self.grid_container, bg=BG)    # Decision (rightmost visually)
        # Pack order controls left→right display: mmWave, CSI, Decision
        self.left_col.pack(side="left", fill="both", expand=True, padx=4)
        self.right_col.pack(side="left", fill="both", expand=True, padx=4)
        self.center_col.pack(side="left", fill="both", expand=True, padx=4)

        tk.Label(self.left_col, text="mmWave  (raw)", fg=BLUE, bg=BG,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=2)
        tk.Label(self.right_col, text="CSI  (raw)", fg=PURPLE, bg=BG,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=2)
        tk.Label(self.center_col, text="Assessment & Analysis  ►►", fg=GREEN, bg=BG,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=2)

        LR = COMPACT_NAMES["BHA2"]  # "Living Room"
        # Living Room tab shows ONLY sensors that physically cover this room:
        # BHA2 (mmWave Living Room) + CSI (WiFi C6 pair). FDA2 covers a different
        # room — do NOT include it here, else it will misleadingly colour the
        # Living Room "present" when someone walks into the Bedroom.
        self.rows = {}
        self.rows["presence"] = ThreeColRow(
            self.left_col, self.center_col, self.right_col, "PRESENCE",
            left_labels=[(LR, "—")],
            right_labels=[("CSI", "—")], center_font_size=20)
        self.rows["motion"] = ThreeColRow(
            self.left_col, self.center_col, self.right_col, "MOTION LEVEL",
            left_labels=[(LR, "distance —")],
            right_labels=[("CSI", "—")], center_font_size=20)
        self.rows["breath"] = ThreeColRow(
            self.left_col, self.center_col, self.right_col, "BREATHING",
            left_labels=[(LR, "—")],
            right_labels=[("CSI", "—")], center_font_size=20)
        self.rows["heart"] = ThreeColRow(
            self.left_col, self.center_col, self.right_col, "HEART RATE",
            left_labels=[(LR, "—")],
            right_labels=[("CSI", "— (exp)")], center_font_size=20)
        # NOTE: FALL DETECTION row is intentionally omitted from Living Room
        # tab — fall is only relevant in the Bedroom zone (FDA2 sensor).
        # See the Bedroom tab for the fall detection panel.
        for r in self.rows.values():
            r.pack()

        # -------- Bedroom tab (fall zone: FDA2 only) --------
        self.bedroom_frame = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.bedroom_frame, text="  🛏  Bedroom  ")

        tk.Label(self.bedroom_frame,
                 text="Zone monitored by ceiling-mounted mmWave sensor (≥2.4m).\n"
                      "Move the webcam to face this room's floor area.",
                 fg=DIM, bg=BG, font=("Segoe UI", 10), justify="left"
                 ).pack(anchor="w", padx=8, pady=(12, 8))

        # Two large status panels — presence and fall
        bedroom_panels = tk.Frame(self.bedroom_frame, bg=BG)
        bedroom_panels.pack(fill="x", padx=8, pady=(0, 8))
        bedroom_panels.grid_columnconfigure(0, weight=1)
        bedroom_panels.grid_columnconfigure(1, weight=1)

        self.bed_presence_frame = tk.Frame(bedroom_panels, bg=PANEL_FUSE, bd=0,
                                           highlightthickness=1, highlightbackground=BORDER_FUSE)
        self.bed_presence_frame.grid(row=0, column=0, sticky="nsew", padx=6, ipadx=12, ipady=16)
        tk.Label(self.bed_presence_frame, text=f"PRESENCE  ({DISPLAY_NAMES['FDA2']})",
                 fg=DIM, bg=PANEL_FUSE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        self.bed_presence_val = tk.Label(self.bed_presence_frame, text="—", fg=DIM,
                                          bg=PANEL_FUSE, font=("Segoe UI", 36, "bold"))
        self.bed_presence_val.pack(pady=(0, 8))

        self.bed_fall_frame = tk.Frame(bedroom_panels, bg=PANEL_FUSE, bd=0,
                                       highlightthickness=1, highlightbackground=BORDER_FUSE)
        self.bed_fall_frame.grid(row=0, column=1, sticky="nsew", padx=6, ipadx=12, ipady=16)
        tk.Label(self.bed_fall_frame, text=f"FALL DETECTION  ({DISPLAY_NAMES['FDA2']})",
                 fg=DIM, bg=PANEL_FUSE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        self.bed_fall_val = tk.Label(self.bed_fall_frame, text="—", fg=DIM,
                                     bg=PANEL_FUSE, font=("Segoe UI", 36, "bold"))
        self.bed_fall_val.pack(pady=(0, 8))

        # Info about which sensors are silent in this zone
        tk.Label(self.bedroom_frame,
                 text="(vitals sensor and CSI motion tracker cover the Living Room only "
                      "and are silent for this zone by design)",
                 fg=DIM, bg=BG, font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=8, pady=(4, 8))

        # -------- Events tab (event log — moved out of window bottom) --------
        self.events_frame = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.events_frame, text="  📋  Events  ")
        tk.Label(self.events_frame,
                 text="All sensor state transitions and system events (newest at bottom).",
                 fg=DIM, bg=BG, font=("Segoe UI", 9, "italic")
                 ).pack(anchor="w", padx=8, pady=(8, 4))
        self.log = scrolledtext.ScrolledText(
            self.events_frame, bg="#0a0d12", fg=FG, insertbackground=FG,
            font=("Consolas", 10), wrap="none", bd=0,
        )
        self.log.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        for tag, colour in [("t", DIM), ("k", BLUE), ("green", GREEN), ("red", RED),
                            ("amber", AMBER), ("fda2", "#79c0ff"),
                            ("bha2", "#d2a8ff"), ("csi", "#a5d6ff"),
                            ("sys", "#f0c674")]:
            self.log.tag_config(tag, foreground=colour)

        # === Motion + HR plots — inside the Living Room tab
        #     (both are only meaningful when the sensors covering that room
        #     are actually seeing the user).
        # Note: the "high activity" reference line replaces the old "fall
        # threshold" label — fall detection is only in the Bedroom tab, so
        # in Living Room context this threshold is just "high motion".
        self.plot = LinePlot(
            self.office_frame, "Motion activity (CSI, last 30s)",
            value_prefix="latest",
            low_ref_label="baseline",
            high_ref_label="high activity",
        )
        self.plot.pack(fill="x", pady=(6, 4))

        self.hr_plot = LinePlot(
            self.office_frame,
            "Heart rate history (mmWave Living Room, last 2 min)",
            height=120,
            window_sec=HR_PLOT_WINDOW_SEC,
            low_ref_label="resting (bpm)", low_ref_colour="#2d5a3d",
            high_ref_label="elevated (bpm)", high_ref_colour="#5c2a2a",
            high_ref_text_colour="#e08484",
            value_prefix="latest HR", value_suffix=" bpm",
            y_min_forced=40, y_max_forced=140,
            line_colour="#f85149",
        )
        self.hr_plot.pack(fill="x", pady=(4, 4))

        # (Family App preview is now the PhoneMockup widget in the left column,
        # under the pose heatmap — not a horizontal panel here.)

        # (Motion + HR plots now live inside the Living Room tab — see above.)

        # (Event log now lives inside the third notebook tab — see below.)

    def _on_open_telegram_web(self):
        """Launch Chrome in app-mode with Telegram Web, positioned right of monitor.

        The window uses a dedicated user-data-dir so Telegram login persists
        across sessions and doesn't affect the user's regular Chrome profile.
        """
        self.root.update()
        mx = self.root.winfo_rootx()
        my = self.root.winfo_rooty()
        mw = self.root.winfo_width()
        tx = mx + mw + TELEGRAM_CHROME_GAP
        ty = my
        chrome_candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            # Edge fallback (Chromium too, supports --app)
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        browser = next((p for p in chrome_candidates if os.path.exists(p)), None)
        if browser is None:
            self._log_evt("SYS", "telegram_web_error",
                          "no Chrome or Edge found in standard install paths", "red")
            return
        try:
            os.makedirs(TELEGRAM_CHROME_PROFILE, exist_ok=True)
            subprocess.Popen(
                [
                    browser,
                    f"--app={TELEGRAM_WEB_URL}",
                    f"--window-size={TELEGRAM_CHROME_W},{TELEGRAM_CHROME_H}",
                    f"--window-position={tx},{ty}",
                    f"--user-data-dir={TELEGRAM_CHROME_PROFILE}",
                ],
                creationflags=(subprocess.DETACHED_PROCESS
                               | subprocess.CREATE_NEW_PROCESS_GROUP),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            self.telegram_web_rect = (tx, ty, TELEGRAM_CHROME_W, TELEGRAM_CHROME_H)
            self._log_evt("SYS", "telegram_web_opened",
                          f"@ {tx},{ty} ({TELEGRAM_CHROME_W}x{TELEGRAM_CHROME_H})",
                          "green")
        except Exception as e:
            self._log_evt("SYS", "telegram_web_error", str(e), "red")

    def _find_telegram_web_window_rect(self):
        """Look up the current REAL screen rect of the Chrome Telegram Web
        window. We prefer to trust the OS over the position we asked Chrome
        for at launch — Chrome may have restored a saved window position or
        the user may have dragged/resized it."""
        try:
            # Search titles that Telegram Web tabs typically show (localised)
            for needle in ("Telegram", "KISUM", "Web A"):
                hits = find_windows_by_title_substring(needle)
                if hits:
                    # Prefer the widest hit (main window, not tooltip)
                    hits.sort(key=lambda t: t[4] * t[5], reverse=True)
                    _, title, x, y, w, h = hits[0]
                    return (x, y, w, h), title
        except Exception:
            pass
        return None, None

    def _on_toggle_rec(self):
        if not self.screen_rec.recording:
            # Monitor-only recording: the Telegram Web window (if open) is
            # intentionally kept out of the frame.
            path = self.screen_rec.start()
            if path is None:
                self.rec_status.config(text="failed to start", fg=RED)
                self._log_evt("SYS", "rec_start_fail", "", "red")
                return
            self.rec_btn.config(text="■ STOP", fg=FG, bg=RED, activebackground="#c53030")
            self.rec_status.config(text=f"→ {os.path.basename(path)}", fg=RED)
            self._log_evt("SYS", "rec_start", path, "red")
        else:
            path = self.screen_rec.stop()
            self.rec_btn.config(text="● REC", fg=RED, bg=PANEL, activebackground="#22272e")
            self.rec_status.config(text=f"saved {os.path.basename(path)}" if path else "stopped", fg=GREEN)
            self._log_evt("SYS", "rec_stop", f"{self.screen_rec.frames_written} frames -> {path}", "green")

    def _on_test_telegram(self):
        """Fire a test alert to Telegram AND to the on-screen preview panel."""
        ok = self.telegram.send(
            "🧪 <b>Test alert from KI SUM AI monitor</b>\n"
            f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "If you see this on your phone, Telegram alerts are working.",
            category="test", cooldown_sec=1,
        )
        self._record_alert_preview("🧪 TEST", "Test alert — verifying Telegram delivery",
                                    GREEN if ok else AMBER)
        if ok:
            self._log_evt("SYS", "telegram_test", "sent", "green")
        else:
            self._log_evt("SYS", "telegram_test",
                          "preview only (Telegram unconfigured / cooldown)", "amber")

    def _on_toggle_log(self):
        if self.log_enabled.get():
            fname = os.path.join(LOG_DIR, f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.log_file = open(fname, "w", newline="", buffering=8192)
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                "iso_time",
                "fda2_person", "fda2_fall",
                "bha2_person", "bha2_breath", "bha2_heart", "bha2_distance", "bha2_targets",
                "csi_present", "csi_motion", "csi_breath_bpm", "csi_heart_bpm",
                "csi_band_breath", "csi_band_slow", "csi_band_walk", "csi_band_fall",
                "csi_baseline_wf",
                "fused_presence", "fused_motion", "fused_breath",
                "fused_heart", "fused_fall",
            ])
            self.log_status.config(text=f"→ {os.path.basename(fname)}", fg=GREEN)
            self._log_evt("SYS", "log_start", fname, "green")
        else:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
                self.log_writer = None
            self.log_status.config(text="log stopped", fg=DIM)
            self._log_evt("SYS", "log_stop", "", "amber")

    # ---------------- pumps ----------------

    def _pump(self):
        # CRITICAL: reschedule OUTSIDE the try/except so a per-event handler
        # error can't stop the pump from firing again. Earlier bug (missing
        # LOG_LABEL_NAMES) crashed _log_evt on first "connected" event, which
        # aborted the whole pump and made the UI appear frozen.
        try:
            while True:
                try:
                    kind, label, payload = self.q.get_nowait()
                except queue.Empty:
                    break
                try:
                    if kind == "connected":
                        self.connected[label] = True
                        host = payload.get("host", "?")
                        self.src_info[label].config(text=f"@ {host}")
                        self._log_evt(label, "connected", host, "green")
                    elif kind == "disconnected":
                        self.connected[label] = False
                        self._log_evt(label, "disconnected", str(payload), "red")
                    elif kind == "discover":
                        self._log_evt(label, "discover", str(payload), "amber")
                    elif kind == "heartbeat":
                        self.last_hb[label] = payload
                    elif kind == "state":
                        self._on_state(label, payload)
                    elif kind == "telegram_callback":
                        self._on_telegram_callback(payload)
                except Exception as e:
                    # Per-event isolation: log the error but keep processing
                    # subsequent events + keep the pump alive.
                    print(f"_pump event handler error: kind={kind} label={label}: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
        finally:
            self.root.after(50, self._pump)

    def _on_state(self, label, payload):
        oid, val, ts = payload
        if label == "CSI":
            if oid == "derived":
                self.csi_derived = val
                # Update motion tracking for inactivity alert
                if val and val.get("motion") not in ("STILL", None):
                    self._last_motion_ts = ts
                    self._inactivity_alerted = False
            elif oid == "sample":
                self.plot.add_sample(ts, val,
                                     baseline=self.csi_derived.get("baseline_wf"),
                                     fall_threshold=self.csi_derived.get("fall_threshold_wf"))
            elif oid == "fall_event":
                self.last_fall_csi_ts = ts
                self._log_evt("CSI", "fall_event", "V3.1 FIRED", "red")
        else:
            prev = self.state[label].get(oid)
            self.state[label][oid] = val
            if prev != val and not oid.endswith("illuminance"):
                tag = "green" if val is False else ("red" if oid == "falling_information" and val is True else "amber")
                self._log_evt(label, oid, f"{prev} → {val}", tag)
                # Fire Telegram alert on real fall (FDA2 falling_information False→True)
                if (label == "FDA2" and oid == "falling_information"
                        and val is True and prev is False):
                    self._send_fall_alert(ts)

    def _tick_ui(self):
        now = time.time()
        self.clock_lbl.config(text=time.strftime("%H:%M:%S", time.localtime(now)))
        for lbl in ("FDA2", "BHA2", "CSI", "POSE"):
            if self.connected[lbl]:
                hb_age = now - self.last_hb[lbl] if self.last_hb[lbl] else 999
                self.src_dots[lbl].config(fg=GREEN if hb_age < 5 else AMBER)
            else:
                self.src_dots[lbl].config(fg=RED)

        # Screen recorder live status
        st = self.screen_rec.status()
        if st is not None:
            self.rec_status.config(
                text=f"REC {st['elapsed']:.0f}s  {st['frames']} frames",
                fg=RED,
            )

        csi_st = dict(self.csi_derived) if self.csi_derived else {}
        csi_st["last_fall_ts"] = self.last_fall_csi_ts
        fused = fuse(self.state["FDA2"], self.state["BHA2"], csi_st)

        # ---- Presence (Living Room zone: BHA2 + CSI only, EXCLUDE FDA2) ----
        r = self.rows["presence"]
        lr_bp = self.state["BHA2"].get("person_information")
        lr_cp = csi_st.get("present")
        r.set_left(0, self._bool_txt(lr_bp, "PRESENT", "none"))
        r.set_right(0, "PRESENT" if lr_cp else "none")
        # Zone-scoped fusion: FDA2 covers a different room → not a source here
        if any(x is True for x in (lr_bp, lr_cp)):
            lr_pres = "PRESENT"
        elif any(x is not None for x in (lr_bp, lr_cp)) and all(
                x is False for x in (lr_bp, lr_cp) if x is not None):
            lr_pres = "NO PERSON"
        else:
            lr_pres = "WAITING"
        pcol = "green" if lr_pres == "PRESENT" else ("dim" if lr_pres == "NO PERSON" else "amber")
        r.set_center(lr_pres, "mmWave or CSI", pcol)

        # ---- Motion ----
        r = self.rows["motion"]
        dist = self.state["BHA2"].get("distance_to_detection_object")
        r.set_left(0, f"distance {dist:.0f}cm" if dist is not None else "distance —")
        r.set_right(0, csi_st.get("motion", "—"))
        r.set_center(fused["motion"], "CSI", "green")

        # ---- Breath ----
        r = self.rows["breath"]
        b_br = self.state["BHA2"].get("real-time_respiratory_rate")
        # Hide out-of-range values — better to show "—" than a misleading "1/min"
        if b_br is not None and BHA2_BREATH_MIN <= b_br <= BHA2_BREATH_MAX:
            r.set_left(0, f"{b_br:.0f}/min", colour="fg")
        else:
            r.set_left(0, "—", colour="dim")
        csi_br = csi_st.get("breath_bpm")
        r.set_right(0, f"~{csi_br:.0f}/min" if csi_br is not None else "—")
        r.set_center(fused["breath"], fused["breath_source"], fused["breath_colour"])

        # ---- Heart ----
        r = self.rows["heart"]
        b_hr = self.state["BHA2"].get("real-time_heart_rate")
        # Same policy: hide out-of-range readings (avoid alarming false values)
        if b_hr is not None and BHA2_HEART_MIN <= b_hr <= BHA2_HEART_MAX:
            r.set_left(0, f"{b_hr:.0f} bpm", colour="fg")
        else:
            r.set_left(0, "—", colour="dim")
        csi_hr_val = csi_st.get("heart_bpm")
        r.set_right(0, f"~{csi_hr_val:.0f} bpm (exp)" if csi_hr_val is not None else "— (exp)")
        r.set_center(fused["heart"], fused["heart_source"], fused["heart_colour"])

        # NOTE: no FALL row in Living Room tab — see Bedroom tab for that.
        # We still need f_f for the Bedroom tab mirror below.
        f_f = self.state["FDA2"].get("falling_information")

        # ---- Bedroom tab (mirror FDA2 presence + fall to big panels) ----
        f_p_val = self.state["FDA2"].get("person_information")
        if f_p_val is True:
            self.bed_presence_val.config(text="PRESENT", fg=GREEN)
        elif f_p_val is False:
            self.bed_presence_val.config(text="none", fg=DIM)
        else:
            self.bed_presence_val.config(text="—", fg=DIM)
        if f_f is True:
            self.bed_fall_val.config(text="FALL DETECTED", fg=RED)
        elif f_f is False:
            self.bed_fall_val.config(text="ok", fg=GREEN)
        else:
            self.bed_fall_val.config(text="—", fg=DIM)

        # ---- Push HR sample if valid ----
        b_hr = self.state["BHA2"].get("real-time_heart_rate")
        if b_hr is not None and BHA2_HEART_MIN <= b_hr <= BHA2_HEART_MAX:
            self.hr_plot.add_sample(now, float(b_hr),
                                    low_ref=70, high_ref=100)

        # ---- Inactivity alert: someone present but no motion for a long time ----
        any_present = (
            self.state["FDA2"].get("person_information") is True
            or self.state["BHA2"].get("person_information") is True
            or (self.csi_derived and self.csi_derived.get("present") is True)
        )
        if any_present and not self._inactivity_alerted:
            idle = now - self._last_motion_ts
            if idle >= INACTIVITY_ALERT_SEC:
                self._send_inactivity_alert(idle)
                self._inactivity_alerted = True
        if not any_present:
            # Reset motion clock so we don't alarm when the person comes back
            self._last_motion_ts = now
            self._inactivity_alerted = False

        # ---- Refresh Family App preview panel ----
        self._refresh_alerts_panel(now)

        # ---- Plot redraw ----
        self.plot.redraw(now)
        self.hr_plot.redraw(now)

        # ---- CSV log ----
        if self.log_writer and now - self.log_last_write > 1.0:
            self.log_last_write = now
            bands = csi_st.get("bands", {}) or {}
            try:
                self.log_writer.writerow([
                    datetime.now().isoformat(timespec="milliseconds"),
                    self.state["FDA2"].get("person_information"),
                    self.state["FDA2"].get("falling_information"),
                    self.state["BHA2"].get("person_information"),
                    self.state["BHA2"].get("real-time_respiratory_rate"),
                    self.state["BHA2"].get("real-time_heart_rate"),
                    self.state["BHA2"].get("distance_to_detection_object"),
                    self.state["BHA2"].get("target_number"),
                    csi_st.get("present"),
                    csi_st.get("motion"),
                    csi_st.get("breath_bpm"),
                    csi_st.get("heart_bpm"),
                    bands.get("breath"), bands.get("slow"), bands.get("walk"), bands.get("fall"),
                    csi_st.get("baseline_wf"),
                    fused["presence"], fused["motion"], fused["breath"],
                    fused["heart"], fused["fall"],
                ])
                self.log_file.flush()
            except Exception:
                pass

        self.root.after(200, self._tick_ui)

    def _record_alert_preview(self, category, text_short, colour):
        """Add an alert to the on-screen Family App preview panel."""
        self.alerts_history.appendleft((time.time(), category, text_short, colour))
        self._latest_alert_ts = time.time()

    def _send_fall_alert(self, ts):
        """Notify family via Telegram + record in preview panel.

        Includes inline keyboard so the family caregiver can respond with
        one tap on their phone: acknowledge / call emergency / call user.
        """
        when = time.strftime("%H:%M:%S", time.localtime(ts))
        preview_text = f"Fall detected in {DISPLAY_NAMES['FDA2']}. Please check immediately."
        telegram_text = (
            f"🚨 <b>Fall detected</b>\n"
            f"Zone: {DISPLAY_NAMES['FDA2']}\n"
            f"Time: {when}\n\n"
            f"Please check on the person immediately."
        )
        keyboard = [
            [
                {"text": "✅ Acknowledged", "callback_data": "ack"},
                {"text": "📞 Call User", "callback_data": "call_user"},
            ],
            [
                {"text": "🆘 Call Emergency (112)", "callback_data": "emergency"},
            ],
        ]
        ok = self.telegram.send(telegram_text, category="fall",
                                cooldown_sec=FALL_ALERT_COOLDOWN_SEC,
                                inline_keyboard=keyboard)
        self._record_alert_preview("🚨 FALL", preview_text, RED)
        self._log_evt("SYS", "telegram_fall",
                      "sent" if ok else "preview only (Telegram unconfigured)",
                      "red" if ok else "amber")

    def _on_telegram_callback(self, cb):
        """Handle inline-button press from the family on their phone."""
        data = cb.get("data", "")
        user = cb.get("from", {}).get("first_name", "family")
        action_map = {
            "ack":       ("✅ ACK",        f"Acknowledged by {user}",       GREEN),
            "call_user": ("📞 CALLING",    f"{user} is calling the user",   AMBER),
            "emergency": ("🆘 EMERGENCY",  f"{user} called Emergency 112",  RED),
        }
        title, msg, colour = action_map.get(
            data, ("👤 REPLY", f"{user}: {data}", BLUE)
        )
        self._record_alert_preview(title, msg, colour)
        self._log_evt("SYS", f"telegram_reply_{data}", user,
                      "green" if data == "ack" else "amber")

    def _send_inactivity_alert(self, seconds_inactive):
        preview_text = f"No motion for {int(seconds_inactive)}s while person present."
        telegram_text = (
            f"⚠️ <b>Prolonged inactivity</b>\n"
            f"No motion detected for {int(seconds_inactive)}s while person present.\n"
            f"Time: {time.strftime('%H:%M:%S')}"
        )
        ok = self.telegram.send(telegram_text, category="inactivity",
                                cooldown_sec=FALL_ALERT_COOLDOWN_SEC * 2)
        self._record_alert_preview("⚠️ INACTIVITY", preview_text, AMBER)
        self._log_evt("SYS", "telegram_inactivity",
                      "sent" if ok else "preview only", "amber")

    def _refresh_alerts_panel(self, now):
        """Push latest alerts into the PhoneMockup widget."""
        self.phone_mockup.update_alerts(list(self.alerts_history), now)

    def _bool_txt(self, val, on, off):
        if val is True:
            return on
        if val is False:
            return off
        return "—"

    _LOG_MAX_LINES = 2000  # cap widget growth so Tk stays responsive

    def _log_evt(self, label, key, msg, tag):
        ts = time.strftime("%H:%M:%S", time.localtime())
        # Use human-readable sensor names (never expose internal FDA2/BHA2 codes)
        display = LOG_LABEL_NAMES.get(label, label)
        tag_lbl = {"FDA2": "fda2", "BHA2": "bha2", "CSI": "csi", "POSE": "csi",
                   "SYS": "sys"}.get(label, "k")
        self.log.insert("end", f"[{ts}] ", "t")
        self.log.insert("end", f"[{display:14}] ", tag_lbl)
        self.log.insert("end", f"{key:28}", "k")
        self.log.insert("end", f"  {msg}\n", tag)
        # Trim oldest lines to keep the widget fast — Tk ScrolledText slows
        # noticeably past a few thousand lines, which can back up the Tk
        # mainloop and delay other callbacks (plot redraw, pump).
        try:
            line_count = int(self.log.index("end-1c").split(".")[0])
            if line_count > self._LOG_MAX_LINES:
                # delete oldest 20% at once (cheaper than one line at a time)
                cut_to = line_count - int(self._LOG_MAX_LINES * 0.8)
                self.log.delete("1.0", f"{cut_to}.0")
        except Exception:
            pass
        self.log.see("end")


# Module-level flag toggled by SIGINT (Ctrl+C); polled by App._check_sigint.
_SIGINT_RECEIVED = [False]


def _install_sigint_handler():
    def _handler(signum, frame):
        _SIGINT_RECEIVED[0] = True
    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass  # not all environments allow custom signal handlers (e.g. non-main-thread)


def parse_args(argv=None):
    """Command-line overrides for the site-specific config at the top of the
    module. Each flag falls back to an environment variable of the same name."""
    p = argparse.ArgumentParser(
        description="Fusion dashboard: Seeed MR60FDA2 + MR60BHA2 (ESPHome API) "
                    "+ ESP32-C6 WiFi CSI (serial).")
    p.add_argument("--csi-port", default=CSI_PORT,
                   help="serial port of the CSI receiver C6, e.g. COM5 (env CSI_PORT)")
    p.add_argument("--fda2-host", default=DEVICES[0]["mdns"],
                   help="MR60FDA2 mDNS hostname, e.g. seeedstudio-mr60fda2-kit-xxxxxx.local "
                        "(env FDA2_HOST)")
    p.add_argument("--fda2-ip", default=DEVICES[0]["fallback_ip"],
                   help="optional MR60FDA2 fallback IP (env FDA2_IP)")
    p.add_argument("--bha2-host", default=DEVICES[1]["mdns"],
                   help="MR60BHA2 mDNS hostname, e.g. seeedstudio-mr60bha2-kit-xxxxxx.local "
                        "(env BHA2_HOST)")
    p.add_argument("--bha2-ip", default=DEVICES[1]["fallback_ip"],
                   help="optional MR60BHA2 fallback IP (env BHA2_IP)")
    p.add_argument("--lan-prefix", default=LAN_PREFIX,
                   help="/24 prefix scanned when mDNS and fallback IP fail (env LAN_PREFIX)")
    p.add_argument("--no-pose", action="store_true",
                   help="disable webcam + MediaPipe pose panel")
    p.add_argument("--webcam", type=int, default=WEBCAM_INDEX,
                   help="webcam index for the pose panel (default 0)")
    p.add_argument("--telegram-web", action="store_true",
                   help="auto-launch Chrome/Edge with Telegram Web next to the monitor")
    return p.parse_args(argv)


def apply_args(args):
    global CSI_PORT, LAN_PREFIX, DISABLE_POSE, WEBCAM_INDEX, AUTO_OPEN_TELEGRAM_WEB
    CSI_PORT = args.csi_port or ""
    LAN_PREFIX = args.lan_prefix
    DEVICES[0]["mdns"] = args.fda2_host or ""
    DEVICES[0]["fallback_ip"] = args.fda2_ip or None
    DEVICES[1]["mdns"] = args.bha2_host or ""
    DEVICES[1]["fallback_ip"] = args.bha2_ip or None
    DISABLE_POSE = DISABLE_POSE or args.no_pose
    WEBCAM_INDEX = args.webcam
    AUTO_OPEN_TELEGRAM_WEB = AUTO_OPEN_TELEGRAM_WEB or args.telegram_web


def main():
    apply_args(parse_args())
    _install_sigint_handler()
    root = tk.Tk()
    App(root)
    try:
        root.mainloop()
    finally:
        # Safety net: force process exit so daemon threads blocked in C-level
        # calls (cv2.VideoCapture.read, serial.read, aioesphomeapi internals)
        # can't hold the process open after the UI is gone.
        os._exit(0)


if __name__ == "__main__":
    main()
