#!/usr/bin/env python3
"""
Kisum Doppler labeled-data collector.

Interactive CLI for capturing Doppler band time-series under labeled scenarios,
so we can tune a fall classifier from real data (not gut feel).

Critical distinctions we're trying to catch:
  - lying down to sleep  vs.  falling to floor
  - sitting down          vs.  falling to floor
  - walking              vs.  everything else

Run this INSTEAD of kisum_radar_monitor.py (both need the same serial port).

Usage: python record_labeled_data.py --port COM5
"""
import argparse
import sys, os, time, csv, threading
from datetime import datetime
import numpy as np
import serial

# Reuse pipeline from the monitor to guarantee identical signal processing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kisum_radar_monitor import (
    SERIAL_PORT, SERIAL_BAUD, parse_csi_line,
    DopplerAnalyzer, DOPPLER_BANDS,
)

# ------------------------- Scenarios -------------------------
# (label, duration_sec, position_hint, instruction)
SCENARIOS = [
    ('baseline_still',    30,
     'Start position: sitting on a chair',
     'Sit on the chair completely still, breathing normally, for 30 s.\n'
     '     This is the baseline reference: person present but motionless.'),

    ('walking',           15,
     'Start position: standing between the two C6 boards',
     'Walk back and forth between the two C6 boards (horizontally) for 15 s.\n'
     '     Natural steps, normal arm swing.'),

    ('sitting_down',      15,
     'Start position: standing, chair 1-2 steps in front of you',
     'From standing -> walk 1-2 steps to the chair -> sit down slowly -> stay still.\n'
     '     About 15 s in total; the last 5-8 s must be motionless.'),

    ('lying_down_sleep',  30,
     'Start position: standing next to the bed',
     '* IMPORTANT *\n'
     '     From standing next to the bed -> lie down slowly (as if going to sleep) ->\n'
     '     stay completely still, breathing normally.\n'
     '     About 30 s in total. This is the "sleep" pattern to compare against falls.'),

    ('fall_simulate',     15,
     'Start position: open floor area (no hard objects nearby; use a mat)',
     '!! BE SAFE !!  * CORE TEST *\n'
     '     From standing -> suddenly "collapse" to the floor (a fast squat-sit,\n'
     '     or squat + hands on the floor, is enough to simulate it).\n'
     '     The motion should be fast and vertical, not a walk.\n'
     '     Afterwards stay lying or crouched, completely still, for at least 8 s.'),
]

CSV_HEADERS = [
    'iso_time', 'scenario', 'phase',
    'band_breath', 'band_slow', 'band_walk', 'band_fall',
    'total_energy', 'fall_ratio',
    'n_subcarriers', 'aggregate_amp',
]


class Collector:
    def __init__(self, out_path, port):
        self.ser = serial.Serial(port, SERIAL_BAUD, timeout=0.01)
        self.doppler = DopplerAnalyzer()
        self.buf = ''
        self.current_scenario = 'idle'
        self.samples_written = 0
        self.out_f = open(out_path, 'w', newline='', buffering=8192)
        self.out_w = csv.writer(self.out_f)
        self.out_w.writerow(CSV_HEADERS)
        self.stop_flag = False
        self._t = threading.Thread(target=self._read_loop, daemon=True)
        self._t.start()

    def _read_loop(self):
        while not self.stop_flag:
            try:
                raw = self.ser.read(self.ser.in_waiting or 1)
            except Exception as e:
                print(f'\nserial err: {e}', flush=True)
                break
            if not raw:
                time.sleep(0.005)
                continue
            try:
                self.buf += raw.decode('utf-8', errors='ignore')
            except Exception:
                continue
            while '\n' in self.buf:
                line, self.buf = self.buf.split('\n', 1)
                csi = parse_csi_line(line)
                if csi is None:
                    continue
                self.doppler.push(csi)
                now_ts = datetime.now().timestamp()
                if self.doppler.maybe_analyze(now_ts):
                    e = self.doppler.band_energy
                    total = e['breath'] + e['slow'] + e['walk'] + e['fall']
                    ratio = e['fall'] / (total + 1e-9)
                    self.out_w.writerow([
                        datetime.fromtimestamp(now_ts).isoformat(timespec='milliseconds'),
                        self.current_scenario,
                        'recording' if self.current_scenario != 'idle' else 'idle',
                        f'{e["breath"]:.2f}',
                        f'{e["slow"]:.2f}',
                        f'{e["walk"]:.2f}',
                        f'{e["fall"]:.2f}',
                        f'{total:.2f}',
                        f'{ratio:.4f}',
                        self.doppler.n_subcarriers,
                        f'{float(np.mean(csi["amplitude"])):.2f}',
                    ])
                    self.samples_written += 1

    def set_scenario(self, name):
        self.current_scenario = name

    def flush(self):
        self.out_f.flush()

    def close(self):
        self.stop_flag = True
        time.sleep(0.3)
        self.out_f.close()
        try:
            self.ser.close()
        except Exception:
            pass


def countdown_bar(seconds):
    """Show a live progress bar during recording."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        left = seconds - elapsed
        if left <= 0:
            break
        bar_full = 30
        filled = int(bar_full * elapsed / seconds)
        bar = '█' * filled + '·' * (bar_full - filled)
        print(f'\r  recording [{bar}] {left:5.1f} s left', end='', flush=True)
        time.sleep(0.2)
    print(f'\r  recording [{"█"*30}] done!            ', flush=True)


def prep_countdown(seconds):
    """Countdown BEFORE recording so user can walk to position."""
    print(f'  → get ready, {seconds} s countdown ...')
    for i in range(int(seconds), 0, -1):
        if i <= 3:
            print(f'    {i} ...', flush=True)
        else:
            print(f'    {i} ...', flush=True)
        time.sleep(1)
    print(f'  → recording started!', flush=True)


def ask_prep_seconds(default=3):
    """Ask user how many seconds they need to get into position."""
    while True:
        raw = input(f' >> Seconds needed to get into position? (Enter = default {default}, or a number like 5): ').strip()
        if raw == '':
            return default
        try:
            n = int(raw)
            if 1 <= n <= 60:
                return n
            print(f'    please enter an integer between 1 and 60')
        except ValueError:
            print(f'    please enter an integer (e.g. 3 or 5)')


def main():
    ap = argparse.ArgumentParser(description='Record 5 labeled CSI Doppler scenarios to CSV.')
    ap.add_argument('--port', default=SERIAL_PORT,
                    help='serial port of the CSI receiver C6, e.g. COM5 (env CSI_PORT)')
    args = ap.parse_args()
    if not args.port:
        ap.error('no serial port given (use --port or set CSI_PORT)')
    port = args.port

    print('=' * 68)
    print(' Kisum Doppler labeled-data recorder')
    print('=' * 68)
    print()
    print(' We will record 5 labeled scenarios in a row.')
    print()
    print(' Flow for each scenario:')
    print('   1. Read the start position and the instructions')
    print('   2. Enter how many seconds you need to get into position (default 3)')
    print('   3. Enter -> countdown -> recording starts -> do the action -> stops automatically')
    print()
    print('   ! During the prep time the CLI counts down 3-2-1; walk to position in time')
    print('   ! Signal collection stops only when the recording time is over')
    print()
    print(' * Prepare 3 positions before recording (same room is fine):')
    print('    1. a chair (to sit down / stand up)')
    print('    2. a bed (to lie down)')
    print('    3. an open floor area (to simulate a fall; no hard objects nearby)')
    print()
    print(' * Keep all positions near the two C6 boards')
    print('   (ideally < 2 m from the line between them)')
    print()
    print(' This data is meant to answer three questions:')
    print('   1. Lying down to sleep vs. falling: can the Doppler signal tell them apart?')
    print('   2. Sitting down slowly vs. falling: can it tell them apart?')
    print('   3. Which band carries the walking energy?')
    print()

    os.makedirs('recordings', exist_ok=True)
    out_path = os.path.join('recordings',
        f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    print(f' Output CSV: {out_path}')
    print()

    print(f' Opening {port} ...')
    try:
        c = Collector(out_path, port)
    except Exception as e:
        print(f' ERROR — cannot open serial port: {e}')
        print(f' Make sure kisum_radar_monitor.py is closed (it also uses {port}).')
        return 1

    print(f' Waiting 2 s for the Doppler pipeline to warm up ...')
    time.sleep(2)
    if c.samples_written == 0:
        print(' WARNING: no Doppler analysis yet — the serial port may have no CSI data.')
        print(' Wait a few seconds, or check that the sender C6 (csi_send) is powered.')
        time.sleep(3)
    print(f' Doppler pipeline running ({c.samples_written} analyses so far)')

    try:
        for i, (name, dur, position_hint, instruction) in enumerate(SCENARIOS):
            print()
            print('=' * 68)
            print(f' Scenario {i+1}/{len(SCENARIOS)}: {name.upper()}   ({dur} s)')
            print('=' * 68)
            print(f' {position_hint}')
            print()
            print(f' Instructions:')
            print(f'     {instruction}')
            print()
            print('-' * 68)
            prep_sec = ask_prep_seconds(default=3)
            prep_countdown(prep_sec)
            c.set_scenario(name)
            before = c.samples_written
            countdown_bar(dur)
            c.set_scenario('idle')
            after = c.samples_written
            c.flush()
            print(f'  recorded "{name}": {after - before} Doppler samples')
            if i < len(SCENARIOS) - 1:
                print(f'  next is "{SCENARIOS[i+1][0]}" — you can walk over while entering the prep time')
            time.sleep(0.5)

        print()
        print('=' * 68)
        print(' All 5 scenarios recorded!')
        print('=' * 68)
        print(f' Total Doppler samples: {c.samples_written}')
        print(f' CSV saved to: {out_path}')
        print()
        print(' Next: compare the 4 bands across the 5 scenarios in this CSV,')
        print('       especially lying_down_sleep vs. fall_simulate.')

    except KeyboardInterrupt:
        print('\n\n Interrupted. The data recorded so far has been saved.')
    finally:
        c.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
