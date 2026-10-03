#!/usr/bin/env python3
"""
Kisum Radar Monitor — lightweight replacement for esp_csi_tool.py (LEGACY,
CSI-only phase of the experiment; superseded by ../unified_monitor.py).

Parses two line types from the patched console_test firmware:
  RADAR_DADA  — Espressif esp-radar wander/jitter outputs (plotted, logged)
  CSI_DATA    — raw CSI → per-subcarrier amplitude → Doppler bands
                (breath / slow / walk / fall) → FallDetectorV2 + stillness panel
Real-time PyQt plot with wall-clock time X-axis and mouse hover.
Per-session CSV logs are written to radar_logs/.

Usage: python kisum_radar_monitor.py --port COM5   (or set env CSI_PORT)
"""
import argparse
import sys, re, os, csv, threading, queue, base64
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QShortcut)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QKeySequence

# ------- Config -------
SERIAL_PORT = os.environ.get('CSI_PORT', '')   # overridden by --port
SERIAL_BAUD = 2000000
WINDOW_SEC = 30                 # visible X-axis span
MAX_POINTS = 4000               # ring buffer size

# Logging: async writer on daemon thread, batched writes, no per-row flush.
# CSV format: iso_time,wander,jitter,wander_th,jitter_th,room,human,event
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radar_logs')
LOG_FLUSH_EVERY_SEC = 2.0       # batch-flush interval (avoid data loss on crash)

# Fall detection based on Power Decline Ratio (PDR).
# The PDR idea was taken from patent literature on WiFi fall detection
# (US 10531817B2, assignee Peking University) and adapted here, independently
# and loosely, to single-antenna esp-radar jitter values. No accuracy figures
# are claimed for this adaptation.
# Core idea: fall produces a SHARP energy decline (before/after ratio >> 1); a
# gentle sit-down produces only a gentle decline. Then verify sustained stillness.
#
# Empirically-tuned thresholds (2026-07-15): a naive PDR>5 triggered on every
# walk-then-sit event. Real walk-sit gives PDR ~20-200 too. What actually
# separates a fall from a controlled sit-down is the PEAK jitter — a fall has
# an impact spike ≥ 0.3, a sit-down peak is closer to 0.15.
FALL_WINDOW_SEC            = 1.0    # size of "before" and "after" energy windows
FALL_PDR_THRESHOLD         = 50.0   # before/after energy ratio > this = candidate
FALL_PEAK_MIN              = 0.25   # peak jitter in the 'before' window must exceed this
FALL_STILLNESS_ENERGY_MAX  = 0.001  # 'after' window energy must be below this
FALL_STILLNESS_SEC         = 5.0    # sustain stillness for this long to confirm
FALL_COOLDOWN_SEC          = 15.0   # min gap between detections

# ------- Doppler config (vertical TX-RX axis) -------
# Physics: when TX and RX are mounted vertically (one high, one low), Doppler shift
# projects onto the vertical velocity component. Walking (horizontal) contributes
# almost nothing; falling / standing / sitting (vertical motion) contribute strongly.
# STFT of the per-subcarrier CSI amplitude time-series yields the Doppler spectrum.
DOPPLER_WINDOW_SEC       = 2.0      # STFT window length
DOPPLER_HOP_SEC          = 0.2      # recompute spectrum every 200 ms
CSI_EXPECTED_RATE_HZ     = 100.0    # firmware csi_recv_interval = 10 ms
DOPPLER_MAX_SUBCARRIERS  = 56       # cap for memory (HT-LTF ~52-56 valid)

# Frequency bands (Hz) → activity type
DOPPLER_BANDS = [
    ('breath',   0.10, 0.60),    # 6-36 BPM breathing
    ('slow',     0.60, 2.00),    # slow body sway, small adjustments
    ('walk',     2.00, 6.00),    # walking, arm swing
    ('fall',     6.00, 25.00),   # sudden vertical motion (fall, jump, sit-down)
]

# ------- CSI_DATA parsing -------
# console_test firmware prints:
#   CSI_DATA,seq,timestamp,collect_num,taget,MAC,rssi,rate,sig_mode,mcs,cwb,0,0,0,
#           stbc,0,0,noise_floor,0,channel,secondary_channel,local_ts,0,0,0,
#           agc_gain,fft_gain,valid_len,0,<base64_bytes>
# The base64 payload is int8 pairs [real0, imag0, real1, imag1, ...] per subcarrier.
CSI_RE = re.compile(
    r'CSI_DATA,\d+,(\d+),\d+,\S*?,[0-9a-fA-F:]+,'
    r'(-?\d+),\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,'
    r'(-?\d+),\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,'
    r'(\d+),\d+,'
    r'([A-Za-z0-9+/=]+)'
)


def parse_csi_line(line):
    """Return dict with wall-time ts, subcarrier amplitude vector (float32), or None."""
    m = CSI_RE.search(line)
    if not m:
        return None
    try:
        fw_ts_ms   = int(m.group(1))
        rssi       = int(m.group(2))
        noise      = int(m.group(3))
        valid_len  = int(m.group(4))
        b64        = m.group(5)
        # base64 → bytes → signed int8 pairs
        pad = (-len(b64)) % 4
        raw = base64.b64decode(b64 + '=' * pad, validate=False)
        arr = np.frombuffer(raw, dtype=np.int8)
        if len(arr) < 4:
            return None
        # Truncate to expected valid_len (firmware may append zeros)
        arr = arr[:min(len(arr), valid_len)]
        # Even length → (real, imag) pairs
        if len(arr) % 2:
            arr = arr[:-1]
        complex_csi = arr[0::2].astype(np.float32) + 1j * arr[1::2].astype(np.float32)
        amplitude = np.abs(complex_csi)  # per-subcarrier magnitude
        if len(amplitude) > DOPPLER_MAX_SUBCARRIERS:
            amplitude = amplitude[:DOPPLER_MAX_SUBCARRIERS]
        return {
            'ts': datetime.now().timestamp(),
            'rssi': rssi,
            'noise': noise,
            'amplitude': amplitude,
        }
    except Exception:
        return None

# ------- Data ring buffers -------
buf = {
    'time':      deque(maxlen=MAX_POINTS),
    'wander':    deque(maxlen=MAX_POINTS),
    'jitter':    deque(maxlen=MAX_POINTS),
    'wander_th': deque(maxlen=MAX_POINTS),
    'jitter_th': deque(maxlen=MAX_POINTS),
    'room':      deque(maxlen=MAX_POINTS),
    'human':     deque(maxlen=MAX_POINTS),
}

RADAR_RE = re.compile(
    r'RADAR_DADA,(\d+),(\d+),'
    r'([-\d.]+),([-\d.]+),([-\d.]+),(\d),'
    r'([-\d.]+),([-\d.]+),([-\d.]+),(\d)'
)


class AsyncLogger:
    """Log rows to CSV on a daemon thread; main loop never blocks on file I/O."""
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        fname = f'radar_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.path = os.path.join(LOG_DIR, fname)
        self.q = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        with open(self.path, 'w', newline='', buffering=8192) as f:
            w = csv.writer(f)
            w.writerow(['iso_time', 'wander', 'jitter',
                        'wander_th', 'jitter_th', 'room', 'human', 'event',
                        'd_breath', 'd_slow', 'd_walk', 'd_fall', 'd_walkfall'])
            last_flush = 0.0
            while not self._stop.is_set() or not self.q.empty():
                try:
                    row = self.q.get(timeout=0.5)
                    w.writerow(row)
                    now = datetime.now().timestamp()
                    if now - last_flush > LOG_FLUSH_EVERY_SEC:
                        f.flush()
                        last_flush = now
                except queue.Empty:
                    f.flush()
                except Exception as e:
                    print(f'AsyncLogger err: {e}')

    def write(self, ts, wander, jitter, wander_th, jitter_th, room, human, event='',
              d_breath=None, d_slow=None, d_walk=None, d_fall=None):
        iso = datetime.fromtimestamp(ts).isoformat(timespec='milliseconds')
        wf = None if (d_walk is None or d_fall is None) else d_walk + d_fall
        self.q.put([iso, wander, jitter, wander_th, jitter_th, room, human, event,
                    d_breath, d_slow, d_walk, d_fall, wf])

    def event(self, tag):
        """Insert a marker (e.g. user pressed SPACE to mark a simulated fall)."""
        now = datetime.now().timestamp()
        iso = datetime.fromtimestamp(now).isoformat(timespec='milliseconds')
        self.q.put([iso, '', '', '', '', '', '', tag, '', '', '', '', ''])

    def stop(self):
        self._stop.set()


class SerialReader:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.txt = ''

    def poll(self):
        """Return (radar_rows, csi_rows). radar_rows for FallDetector, csi_rows for Doppler."""
        radar_rows = []
        csi_rows   = []
        try:
            raw = self.ser.read(self.ser.in_waiting or 1)
        except Exception as e:
            print(f'serial read err: {e}')
            return radar_rows, csi_rows
        if not raw:
            return radar_rows, csi_rows
        self.txt += raw.decode('utf-8', errors='ignore')
        while '\n' in self.txt:
            line, self.txt = self.txt.split('\n', 1)
            m = RADAR_RE.search(line)
            if m:
                radar_rows.append({
                    'ts': datetime.now().timestamp(),
                    'wander':    float(m.group(3)),
                    'wander_th': float(m.group(5)),
                    'room':      int(m.group(6)),
                    'jitter':    float(m.group(7)),
                    'jitter_th': float(m.group(9)),
                    'human':     int(m.group(10)),
                })
                continue
            csi = parse_csi_line(line)
            if csi is not None:
                csi_rows.append(csi)
        return radar_rows, csi_rows


class DopplerAnalyzer:
    """Compute Doppler spectrum from per-subcarrier CSI amplitude time-series.

    Physics: with the two C6 boards mounted VERTICALLY (one high, one low), the TX-RX
    axis is vertical. Doppler shift measures velocity along that axis:
        walking (horizontal)         → low Doppler (rejected)
        falling / standing / sitting → high Doppler along the vertical axis
        breathing / heartbeat        → very-low-frequency periodic Doppler

    Method:
      1. Ring-buffer per-subcarrier amplitude, sampled at ~100 Hz for DOPPLER_WINDOW_SEC.
      2. Remove DC per subcarrier (subtract mean).
      3. FFT per subcarrier → magnitude spectrum.
      4. Average the magnitude spectra across subcarriers (subcarrier diversity → SNR).
      5. Sum energy in each DOPPLER_BANDS band → activity fingerprint.
    """
    def __init__(self):
        # Ring buffer: list of (ts, amplitude_vector). We resample onto a uniform
        # grid at analysis time.
        self.hist = deque()
        # Latest results
        self.spectrum_freqs = None      # numpy array, Hz
        self.spectrum_power = None      # numpy array, aggregated magnitude
        self.band_energy   = {name: 0.0 for name, _, _ in DOPPLER_BANDS}
        self.n_subcarriers = 0
        self.last_analysis_ts = 0.0
        # Track last-N-seconds peak per band so brief bursts stay visible in UI.
        self.band_peak_5s  = {name: 0.0 for name, _, _ in DOPPLER_BANDS}
        # (ts, {name: val}) history for peak calculation
        self.band_hist = deque()
        # ---- V3.1 Adaptive baseline: rolling walk+fall history for baseline ----
        # We compute p10 over the last 5 min → this is the "room-still" reference.
        self.wf_history_sec = 300.0   # 5 min
        self.wf_history = deque()     # (ts, walk+fall)
        self.baseline_wf = 500.0      # default fallback until we have enough samples
        self.baseline_ready = False   # true after WARMUP_SEC of data
        # ---- Breath-band posture hypothesis (2026-07-17) ----
        # Physics: with vertical C6 axis, chest-motion (breathing) along the vertical
        # axis gives HIGHER Doppler than motion perpendicular to it. Standing chest
        # expansion is horizontal (perpendicular to axis) → weak breath band.
        # Lying-face-up chest expansion is vertical (aligned with axis) → strong
        # breath band. So breath band level, normalised to a rolling baseline,
        # may distinguish posture. We track rolling p50 (median) of breath as
        # baseline.
        self.breath_history = deque()   # (ts, breath_energy)
        self.breath_baseline = 20.0     # default fallback
        self.breath_history_sec = 300.0

    def _trim(self, now_ts):
        cutoff = now_ts - DOPPLER_WINDOW_SEC * 1.2
        while self.hist and self.hist[0][0] < cutoff:
            self.hist.popleft()

    def push(self, csi_row):
        self.hist.append((csi_row['ts'], csi_row['amplitude']))
        self._trim(csi_row['ts'])

    def maybe_analyze(self, now_ts):
        """Recompute spectrum if enough time has passed. Returns True if updated."""
        if now_ts - self.last_analysis_ts < DOPPLER_HOP_SEC:
            return False
        if len(self.hist) < int(CSI_EXPECTED_RATE_HZ * DOPPLER_WINDOW_SEC * 0.5):
            return False
        # Extract time-series
        times  = np.array([t for (t, _) in self.hist], dtype=np.float64)
        amps   = np.stack([a for (_, a) in self.hist])  # (T, S)
        # Align subcarrier count across packets (pad with 0 if one is shorter)
        n_sub  = min(a.shape[0] for a in [x[1] for x in self.hist])
        amps   = amps[:, :n_sub]
        if n_sub < 4:
            return False
        # Resample to uniform grid at CSI_EXPECTED_RATE_HZ
        t0, t1 = times[0], times[-1]
        n_samples = max(16, int((t1 - t0) * CSI_EXPECTED_RATE_HZ))
        if n_samples < 16:
            return False
        t_uniform = np.linspace(t0, t1, n_samples)
        uniform = np.empty((n_samples, n_sub), dtype=np.float32)
        for s in range(n_sub):
            uniform[:, s] = np.interp(t_uniform, times, amps[:, s])
        # Remove DC per subcarrier
        uniform -= uniform.mean(axis=0, keepdims=True)
        # Window (Hann) reduces spectral leakage
        window = np.hanning(n_samples).astype(np.float32)[:, None]
        uniform *= window
        # FFT per subcarrier
        spec = np.abs(np.fft.rfft(uniform, axis=0))  # (F, S)
        # Average across subcarriers → aggregated spectrum
        agg = spec.mean(axis=1)  # (F,)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / CSI_EXPECTED_RATE_HZ)
        # Band energies
        for name, lo, hi in DOPPLER_BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            self.band_energy[name] = float(agg[mask].sum()) if mask.any() else 0.0
        self.spectrum_freqs = freqs
        self.spectrum_power = agg
        self.n_subcarriers = n_sub
        self.last_analysis_ts = now_ts
        # Update band-peak-over-5-sec
        self.band_hist.append((now_ts, dict(self.band_energy)))
        cutoff = now_ts - 5.0
        while self.band_hist and self.band_hist[0][0] < cutoff:
            self.band_hist.popleft()
        for name, _, _ in DOPPLER_BANDS:
            self.band_peak_5s[name] = max(d[name] for _, d in self.band_hist)
        # ---- V3.1 Adaptive baseline update ----
        wf_now = self.band_energy['walk'] + self.band_energy['fall']
        self.wf_history.append((now_ts, wf_now))
        cutoff = now_ts - self.wf_history_sec
        while self.wf_history and self.wf_history[0][0] < cutoff:
            self.wf_history.popleft()
        # Need at least 30 sec of samples for a stable baseline
        if len(self.wf_history) > 100:
            vals = sorted(v for _, v in self.wf_history)
            self.baseline_wf = float(vals[len(vals) // 10])  # p10
            self.baseline_ready = True
        # Track breath baseline separately (rolling p50 median)
        breath_now = self.band_energy['breath']
        self.breath_history.append((now_ts, breath_now))
        cutoff = now_ts - self.breath_history_sec
        while self.breath_history and self.breath_history[0][0] < cutoff:
            self.breath_history.popleft()
        if len(self.breath_history) > 100:
            bvals = sorted(v for _, v in self.breath_history)
            self.breath_baseline = float(bvals[len(bvals) // 2])  # p50
        return True


class FallDetectorV2:
    """Doppler-based fall detector — V3.1 adaptive-baseline edition.

    Thresholds are now RATIOS against an adaptive baseline (rolling p10 of
    walk+fall over the last 5 min) instead of fixed absolute numbers.
    The old fixed thresholds (BURST_MIN=1200 etc.) corresponded to a baseline
    of ~500 in the labeled recording. If your environment's baseline is
    different, the RATIOS still hold — that's the whole point of V3.1.

    Fallback: if the baseline hasn't warmed up yet, we use a static 500 default.
    """
    # Ratios (× baseline). Baseline ≈ 500 in the labeled session, so these
    # multiply to the same effective thresholds V2 stable used.
    BURST_MIN_R         = 2.4    # 2.4 × baseline (~ 1200 if baseline=500)
    BURST_PEAK_MAX_R    = 4.4    # 4.4 × baseline (~ 2200 if baseline=500)
    STILLNESS_MAX_R     = 1.8    # 1.8 × baseline (~ 900 if baseline=500)
    STILLNESS_RANGE_R   = 0.60   # 0.60 × baseline (~ 300 if baseline=500)
    STILLNESS_RANGE_MIN_R = 0.12 # 0.12 × baseline (~ 60 if baseline=500)
    STILLNESS_RESUME    = 1.3    # stillness_max * this = motion-resumed threshold
    STILLNESS_SEC       = 6.0
    BURST_MAX_SEC       = 6.0
    COOLDOWN_SEC        = 15.0
    STILLNESS_RESUME  = 1.3    # stillness_max * this = motion-resumed threshold
    COOLDOWN_SEC      = 15.0

    def __init__(self):
        self.hist = deque()          # (ts, walk+fall)
        self.state = 'idle'          # idle | burst | stillness
        self.burst_ts = None
        self.burst_peak = 0.0        # track the max wf during burst
        self.stillness_start = None
        self.stillness_samples = []
        self.last_alert = -1e9       # so cooldown never blocks initial detection

    def update(self, ts, walk, fall, baseline=500.0):
        """baseline = rolling p10 of walk+fall over the last 5 min (V3.1)."""
        wf = walk + fall
        self.hist.append((ts, wf))
        while self.hist and self.hist[0][0] < ts - 20:
            self.hist.popleft()

        if ts - self.last_alert < self.COOLDOWN_SEC:
            return False

        # Compute effective thresholds from the current baseline
        BURST_MIN         = self.BURST_MIN_R         * baseline
        BURST_PEAK_MAX    = self.BURST_PEAK_MAX_R    * baseline
        STILLNESS_MAX     = self.STILLNESS_MAX_R     * baseline
        STILLNESS_RANGE   = self.STILLNESS_RANGE_R   * baseline
        STILLNESS_RANGE_MIN = self.STILLNESS_RANGE_MIN_R * baseline

        if self.state == 'idle':
            if wf > BURST_MIN:
                self.state = 'burst'
                self.burst_ts = ts
                self.burst_peak = wf

        elif self.state == 'burst':
            if wf > self.burst_peak:
                self.burst_peak = wf
            if wf < STILLNESS_MAX:
                self.state = 'stillness'
                self.stillness_start = ts
                self.stillness_samples = [wf]
            elif ts - self.burst_ts > self.BURST_MAX_SEC:
                self.state = 'idle'
                self.burst_peak = 0.0

        elif self.state == 'stillness':
            self.stillness_samples.append(wf)
            if wf > STILLNESS_MAX * self.STILLNESS_RESUME:
                self.state = 'idle'
                self.burst_peak = 0.0
                return False
            still_range = max(self.stillness_samples) - min(self.stillness_samples)
            if still_range > STILLNESS_RANGE:
                self.state = 'idle'
                self.burst_peak = 0.0
                return False
            if ts - self.stillness_start >= self.STILLNESS_SEC:
                if self.burst_peak > BURST_PEAK_MAX:
                    print(f'[FALL_V2 gated] burst_peak={self.burst_peak:.0f} > {BURST_PEAK_MAX:.0f} (baseline={baseline:.0f}) = walking/leaving')
                    self.state = 'idle'
                    self.burst_peak = 0.0
                    return False
                if still_range < STILLNESS_RANGE_MIN:
                    print(f'[FALL_V2 gated] still_range={still_range:.0f} < {STILLNESS_RANGE_MIN:.0f} (baseline={baseline:.0f}) = empty room')
                    self.state = 'idle'
                    self.burst_peak = 0.0
                    return False
                self.last_alert = ts
                self.state = 'idle'
                dt = ts - self.burst_ts if self.burst_ts else 0
                print(f'[FALL_V2] {datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")}  '
                      f'burst_dur={dt:.1f}s  peak={self.burst_peak:.0f}  still_range={still_range:.0f}  baseline={baseline:.0f}')
                self.burst_peak = 0.0
                return True
        return False


class FallDetector:
    """Power-Decline-Ratio (PDR) fall detector.

    Method (loosely adapted from the PDR idea in US 10531817B2, single antenna):
      1. Keep a sliding buffer of (ts, jitter^2) covering the last ~7 sec.
      2. Every tick compute PDR = mean_energy_before / mean_energy_after,
         where 'before' and 'after' are two 1-sec windows straddling a
         reference point ~1 sec in the past.
      3. If PDR > 5 AND after-window energy < FALL_STILLNESS_ENERGY_MAX,
         we have a fall candidate anchored at that reference point.
      4. Then verify the after-window stays quiet for FALL_STILLNESS_SEC
         (5 sec total). If so → FALL.
    Rejects "walk then sit" because that transition is gradual (PDR is low).
    """
    def __init__(self):
        self.hist = deque()   # (ts, jitter) — raw, not squared
        self.candidate_ref_ts = None  # anchor time (~1 sec ago at detection)
        self.candidate_pdr = 0.0
        self.candidate_peak = 0.0
        self.last_alert_ts = 0.0

    def _trim(self, now_ts):
        # keep enough history for before-window(1s) + stillness(5s) + margin
        cutoff = now_ts - (FALL_WINDOW_SEC + FALL_STILLNESS_SEC + 2.0)
        while self.hist and self.hist[0][0] < cutoff:
            self.hist.popleft()

    def _window_stats(self, t0, t1):
        """Return (mean_energy, peak_jitter) over the interval, or (None, 0)."""
        vals = [j for (ts, j) in self.hist if t0 <= ts <= t1]
        if not vals:
            return None, 0.0
        mean_e = sum(j * j for j in vals) / len(vals)
        return mean_e, max(vals)

    def update(self, ts, jitter):
        self.hist.append((ts, jitter))
        self._trim(ts)

        if ts - self.last_alert_ts < FALL_COOLDOWN_SEC:
            return False

        # ---- Detect new candidate (only if we don't already have one) ----
        if self.candidate_ref_ts is None:
            ref = ts - FALL_WINDOW_SEC              # ~1 sec in the past
            e_before, peak_before = self._window_stats(ref - FALL_WINDOW_SEC, ref)
            e_after,  _           = self._window_stats(ref, ref + FALL_WINDOW_SEC)
            if e_before is None or e_after is None:
                return False
            pdr = e_before / (e_after + 1e-9)
            if (pdr >= FALL_PDR_THRESHOLD
                    and peak_before >= FALL_PEAK_MIN
                    and e_after <= FALL_STILLNESS_ENERGY_MAX):
                self.candidate_ref_ts = ref
                self.candidate_pdr = pdr
                self.candidate_peak = peak_before
            return False

        # ---- We have a candidate: verify sustained stillness ----
        e_since, _ = self._window_stats(self.candidate_ref_ts, ts)
        if e_since is None:
            return False
        if e_since > FALL_STILLNESS_ENERGY_MAX * 2:
            self.candidate_ref_ts = None
            self.candidate_pdr = 0.0
            self.candidate_peak = 0.0
            return False
        if ts - self.candidate_ref_ts >= FALL_STILLNESS_SEC:
            self.last_alert_ts = ts
            pdr = self.candidate_pdr; peak = self.candidate_peak
            self.candidate_ref_ts = None
            self.candidate_pdr = 0.0
            self.candidate_peak = 0.0
            print(f'[FALL] {datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")}  PDR={pdr:.1f}  peak={peak:.3f}')
            return True
        return False


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                out.append(datetime.fromtimestamp(v).strftime('%H:%M:%S'))
            except Exception:
                out.append('')
        return out


class Monitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'Kisum Radar Monitor — {SERIAL_PORT}')
        self.resize(1500, 800)

        self.reader = SerialReader(SERIAL_PORT, SERIAL_BAUD)
        self.fall = FallDetector()          # OLD PDR-based (disabled in _tick, kept for reference)
        self.fall_v2 = FallDetectorV2()     # NEW Doppler-based (data-driven from 2026-07-17)
        self.doppler = DopplerAnalyzer()
        self.logger = AsyncLogger()
        self.fall_count = 0
        self.last_fall = None
        self.fall_markers = []
        self.last_state = 'STARTING'
        self.last_state_change_ts = 0.0
        # Hysteresis: pending state must persist this long before we display it
        self.pending_state = 'STARTING'
        self.pending_state_since = 0.0

        # SPACE key marks a moment in the log (e.g. "I just simulated a fall")
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._mark_event)
        # F key marks "actual fall attempt now"
        QShortcut(QKeySequence(Qt.Key_F), self, activated=lambda: self._mark_event('FALL_MARK'))
        # W key marks "walking to sit" false-positive scenario
        QShortcut(QKeySequence(Qt.Key_W), self, activated=lambda: self._mark_event('WALK_SIT'))

        # --- layout ---
        central = QWidget(); self.setCentralWidget(central)
        v = QVBoxLayout(central)

        self.status_label = QLabel('WAITING FOR DATA...')
        self.status_label.setFont(QFont('Arial', 26, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            'background: #222; color: white; padding: 18px;')
        v.addWidget(self.status_label)

        mid = QWidget(); mid_h = QHBoxLayout(mid); v.addWidget(mid, 1)

        # plot (with time axis)
        self.plot = pg.PlotWidget(axisItems={'bottom': TimeAxis(orientation='bottom')})
        self.plot.setBackground('k')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 'Wall clock time')
        self.plot.addLegend()

        self.c_wander    = self.plot.plot(pen=pg.mkPen((255,100,255), width=2),
                                          name='wander (presence?)')
        self.c_jitter    = self.plot.plot(pen=pg.mkPen((100,255,100), width=2),
                                          name='jitter (motion?)')
        self.c_wander_th = self.plot.plot(pen=pg.mkPen((255,255,255), width=1,
                                                        style=Qt.DashLine),
                                          name='wander threshold')
        self.c_jitter_th = self.plot.plot(pen=pg.mkPen((100,150,255), width=1,
                                                        style=Qt.DashLine),
                                          name='jitter threshold')

        # crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen('y', width=1))
        self.hLine = pg.InfiniteLine(angle=0,  movable=False,
                                     pen=pg.mkPen('y', width=1))
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.plot.scene().sigMouseMoved.connect(self._hover)

        mid_h.addWidget(self.plot, 3)

        info = QWidget(); iv = QVBoxLayout(info)
        self.hover_label = QLabel(
            'Move mouse over the plot for a point read-out.')
        self.hover_label.setFont(QFont('Consolas', 12))
        self.hover_label.setStyleSheet(
            'background:#111;color:#0f0;padding:15px;')
        self.hover_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.hover_label.setMinimumWidth(320)
        iv.addWidget(self.hover_label)

        self.fall_label = QLabel('Falls detected: 0\nLast fall: -')
        self.fall_label.setFont(QFont('Arial', 14, QFont.Bold))
        self.fall_label.setStyleSheet(
            'background:#300;color:white;padding:15px;')
        self.fall_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        iv.addWidget(self.fall_label)

        # Live human-state indicator
        self.state_label = QLabel('STATE:  ...')
        self.state_label.setFont(QFont('Arial', 20, QFont.Bold))
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setStyleSheet(
            'background:#222;color:#fff;padding:15px;')
        self.state_label.setMinimumWidth(320)
        iv.addWidget(self.state_label)

        # POSTURE panel — experimental breath-based classifier
        # Hypothesis (2026-07-17): with vertical C6 axis, lying breath is stronger
        # than standing breath because breath motion aligns with the Doppler axis.
        self.posture_label = QLabel('POSTURE:  ...')
        self.posture_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.posture_label.setAlignment(Qt.AlignCenter)
        self.posture_label.setStyleSheet(
            'background:#334455;color:#fff;padding:12px;')
        self.posture_label.setMinimumWidth(320)
        iv.addWidget(self.posture_label)

        # Doppler band energy panel (real-time indicator of motion type)
        self.doppler_label = QLabel(
            'Doppler bands — waiting for data...')
        self.doppler_label.setFont(QFont('Consolas', 11))
        self.doppler_label.setStyleSheet(
            'background:#001030;color:#8ff;padding:15px;')
        self.doppler_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        iv.addWidget(self.doppler_label)
        mid_h.addWidget(info, 1)

        # Second row: Doppler spectrum plot
        self.dop_plot = pg.PlotWidget()
        self.dop_plot.setBackground('k')
        self.dop_plot.showGrid(x=True, y=True, alpha=0.3)
        self.dop_plot.setLabel('left', 'Amplitude')
        self.dop_plot.setLabel('bottom', 'Doppler frequency (Hz)')
        self.dop_plot.addLegend()
        self.dop_plot.setXRange(0, 30, padding=0)
        self.dop_plot.setMaximumHeight(260)
        self.c_doppler = self.dop_plot.plot(pen=pg.mkPen((100,220,255), width=2),
                                             name='Doppler spectrum (mean over subcarriers)')
        # Band boundary vertical dashed lines
        for name, lo, hi in DOPPLER_BANDS:
            for x in (lo, hi):
                self.dop_plot.addItem(pg.InfiniteLine(
                    pos=x, angle=90, movable=False,
                    pen=pg.mkPen((80,80,80), width=1, style=Qt.DashLine)))
        v.addWidget(self.dop_plot)

        self.timer = QTimer(); self.timer.timeout.connect(self._tick)
        self.timer.start(50)  # 20 Hz refresh

    def _hover(self, pos):
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        mp = self.plot.getViewBox().mapSceneToView(pos)
        self.vLine.setPos(mp.x()); self.hLine.setPos(mp.y())
        if not buf['time']:
            return
        arr = np.array(buf['time'])
        idx = int(np.argmin(np.abs(arr - mp.x())))
        ts = buf['time'][idx]
        dt = datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]
        lat = (datetime.now().timestamp() - ts) * 1000
        room = 'someone' if buf['room'][idx] else 'none'
        human= 'move'    if buf['human'][idx] else 'static'
        self.hover_label.setText(
            f'Point time:  {dt}\n'
            f'Latency:     {lat:.0f} ms\n'
            f'\n'
            f'wander:      {buf["wander"][idx]:.5f}\n'
            f'wander_th:   {buf["wander_th"][idx]:.5f}\n'
            f'jitter:      {buf["jitter"][idx]:.5f}\n'
            f'jitter_th:   {buf["jitter_th"][idx]:.5f}\n'
            f'\n'
            f'Room:        {room}\n'
            f'Human:       {human}\n'
        )

    def _mark_event(self, tag='MARK'):
        self.logger.event(tag)
        self.status_label.setText(self.status_label.text() + f'  [{tag}]')

    # State indicator — REVISED 2026-07-17: uses breath band (0.1-0.6 Hz),
    # same signal as the STILLNESS panel below. In live testing this was the
    # most reliable "moving vs still" indicator we have on single-antenna CSI.
    # Rationale: walk+fall band (2-25 Hz) needs FAST motion; slow motion
    # (limb adjust, standing up slowly) doesn't trigger it — false STILL.
    # Breath band catches ANY 0.1-0.6 Hz motion including slow movement.
    STATE_MOVE_MIN     = 30.0   # MOVING if breath >= 30 (matches STILLNESS "SMALL MOVEMENTS")
    STATE_EMPTY_MAX_R  = 0.6    # LIKELY_EMPTY if wf < 0.6 × baseline

    def _update_state_label(self, ts, e):
        breath   = e['breath']                    # INSTANTANEOUS
        wf       = e['walk'] + e['fall']
        baseline = self.doppler.baseline_wf
        empty_max = self.STATE_EMPTY_MAX_R * baseline
        recent_fall_sec = (ts - self.fall_v2.last_alert) if self.fall_v2.last_alert > 0 else 1e9

        raw_moving = breath >= self.STATE_MOVE_MIN

        if raw_moving:
            state, color, emoji = 'MOVING',            '#309030', '🚶'
        elif recent_fall_sec < 30:
            state, color, emoji = 'RECENT_FALL_OR_LIE', '#c04040', '🛏️'
        elif wf < empty_max:
            state, color, emoji = 'LIKELY_EMPTY',      '#404060', '🚪'
        else:
            state, color, emoji = 'STILL',             '#c0a030', '🧍'

        if state != self.last_state:
            self.last_state = state
            self.last_state_change_ts = ts
        dur = ts - self.last_state_change_ts
        self.state_label.setStyleSheet(
            f'background:{color};color:#fff;padding:15px;')
        ready = '' if self.doppler.baseline_ready else ' [warmup]'
        self.state_label.setText(
            f'{emoji}  {state.replace("_", " ")}   ({dur:.0f}s)\n'
            f'breath={breath:.0f}  (moving if >= {self.STATE_MOVE_MIN:.0f}){ready}')

    # "Posture" panel — HONESTLY REVISED 2026-07-17 evening.
    # User's leg-movement test proved: breath band (0.1-0.6 Hz) captures ANY
    # slow motion (leg twitch, arm adjust, deep breath) — not just posture.
    # Sit vs lie is INDISTINGUISHABLE on single-antenna CSI.
    #   We CAN say: "very still" vs "small movements" vs "obvious motion"
    #   We CANNOT say: "sitting vs standing vs lying" reliably
    # Show honest STILLNESS LEVEL instead of a fake posture.
    def _update_posture_label(self, ts, e):
        breath = e['breath']

        if not self.doppler.baseline_ready:
            level, color = 'CALIBRATING ...', '#404060'
        elif breath < 10:
            level, color = '💤 VERY STILL (no small movement)', '#204070'
        elif breath < 30:
            level, color = '🪑 STILL (minor adjustments)', '#308060'
        elif breath < 80:
            level, color = '✋ SMALL MOVEMENTS (limb / breath deep)', '#a08030'
        else:
            level, color = '🚴 LARGE MOVEMENTS', '#a03030'

        self.posture_label.setStyleSheet(
            f'background:{color};color:#fff;padding:12px;')
        self.posture_label.setText(
            f'STILLNESS:  {level}\n'
            f'breath={breath:.1f}   (single-antenna CSI cannot distinguish sit/lie)')

    def _tick(self):
        radar_rows, csi_rows = self.reader.poll()
        for r in radar_rows:
            buf['time'].append(r['ts'])
            buf['wander'].append(r['wander'])
            buf['jitter'].append(r['jitter'])
            buf['wander_th'].append(r['wander_th'])
            buf['jitter_th'].append(r['jitter_th'])
            buf['room'].append(r['room'])
            buf['human'].append(r['human'])
            e = self.doppler.band_energy
            self.logger.write(r['ts'], r['wander'], r['jitter'],
                              r['wander_th'], r['jitter_th'],
                              r['room'], r['human'],
                              d_breath=e.get('breath'), d_slow=e.get('slow'),
                              d_walk=e.get('walk'), d_fall=e.get('fall'))
            # OLD jitter-based fall detector — DISABLED while we develop Doppler-based one.
            # Kept for reference / A-B comparison later.
            # if self.fall.update(r['ts'], r['jitter']):
            #     ...

        # Feed CSI rows to Doppler analyzer (independent stream, higher rate)
        for c in csi_rows:
            self.doppler.push(c)

        now_ts = datetime.now().timestamp()
        if self.doppler.maybe_analyze(now_ts):
            # Feed Doppler bands to the new fall detector
            e = self.doppler.band_energy
            if self.fall_v2.update(now_ts, e['walk'], e['fall'],
                                   baseline=self.doppler.baseline_wf):
                self.fall_count += 1
                self.last_fall = datetime.fromtimestamp(now_ts)
                self.logger.event('FALL_V2_DETECTED')
                # add marker on the wander/jitter time plot
                mk = pg.InfiniteLine(pos=now_ts, angle=90,
                                     pen=pg.mkPen('r', width=3))
                self.plot.addItem(mk); self.fall_markers.append(mk)
                while len(self.fall_markers) > 5:
                    self.plot.removeItem(self.fall_markers.pop(0))
            # Update the live state indicator (uses INSTANTANEOUS band values, no 5s peak)
            self._update_state_label(now_ts, e)
            # Update posture panel (breath-band-based hypothesis)
            self._update_posture_label(now_ts, e)
            # New Doppler spectrum available — update plot + band panel
            self.c_doppler.setData(self.doppler.spectrum_freqs,
                                   self.doppler.spectrum_power)
            e = self.doppler.band_energy
            p = self.doppler.band_peak_5s
            total = sum(e.values()) + 1e-9
            def bar(val, ref):
                n = int(min(15, max(0, val / (ref + 1e-9) * 15)))
                return '█' * n + '·' * (15 - n)
            ref = max(p.values()) + 1e-9   # scale by 5-sec peak
            self.doppler_label.setText(
                f'Doppler bands ({self.doppler.n_subcarriers} subcarriers, '
                f'{len(self.doppler.hist)} samples)\n'
                f'                        NOW    5s-PEAK   bar (now/peak)\n'
                f'breath (0.1-0.6 Hz) : {e["breath"]:7.1f}  {p["breath"]:7.1f}  {bar(e["breath"], ref)}\n'
                f'slow   (0.6-2 Hz)   : {e["slow"]:7.1f}  {p["slow"]:7.1f}  {bar(e["slow"], ref)}\n'
                f'walk   (2-6 Hz)     : {e["walk"]:7.1f}  {p["walk"]:7.1f}  {bar(e["walk"], ref)}\n'
                f'FALL   (6-25 Hz)    : {e["fall"]:7.1f}  {p["fall"]:7.1f}  {bar(e["fall"], ref)}\n\n'
                f'total: {total:.0f}   ratio fall/total: {e["fall"]/total*100:5.1f}%'
            )

        if not buf['time']:
            return

        t = np.array(buf['time'])
        self.c_wander.setData(t,    np.array(buf['wander']))
        self.c_jitter.setData(t,    np.array(buf['jitter']))
        self.c_wander_th.setData(t, np.array(buf['wander_th']))
        self.c_jitter_th.setData(t, np.array(buf['jitter_th']))

        now_ts = datetime.now().timestamp()
        self.plot.setXRange(now_ts - WINDOW_SEC, now_ts, padding=0)

        latest_ts = buf['time'][-1]
        lat = (now_ts - latest_ts) * 1000
        r_last = buf['room'][-1]; h_last = buf['human'][-1]
        room  = 'SOMEONE' if r_last else 'NONE'
        human = 'MOVE'    if h_last else 'STATIC'
        if   r_last and h_last:      bg,fg='#080','white'   # green
        elif r_last and not h_last:  bg,fg='#eee','black'   # white
        elif not r_last and h_last:  bg,fg='#800','white'   # red (weird)
        else:                        bg,fg='#222','#888'    # dim (empty)
        self.status_label.setStyleSheet(
            f'background:{bg};color:{fg};padding:18px;')
        stamp = datetime.fromtimestamp(latest_ts).strftime('%H:%M:%S.%f')[:-3]
        self.status_label.setText(
            f'{room} {human}    |    last: {stamp}    |    latency: {lat:.0f} ms')

        if self.last_fall:
            secs = (datetime.now() - self.last_fall).total_seconds()
            self.fall_label.setText(
                f'Falls detected: {self.fall_count}\n'
                f'Last fall: {self.last_fall.strftime("%H:%M:%S")} '
                f'({secs:.0f} sec ago)')


if __name__ == '__main__':
    _ap = argparse.ArgumentParser(description='Legacy CSI-only monitor (PyQt).')
    _ap.add_argument('--port', default=SERIAL_PORT,
                     help='serial port of the CSI receiver C6, e.g. COM5 (env CSI_PORT)')
    _args = _ap.parse_args()
    if not _args.port:
        _ap.error('no serial port given (use --port or set CSI_PORT)')
    SERIAL_PORT = _args.port
    app = QApplication(sys.argv)
    w = Monitor(); w.show()
    sys.exit(app.exec_())
