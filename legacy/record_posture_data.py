#!/usr/bin/env python3
"""
Kisum Posture Data Collector.

Goal: test the "breathing direction -> posture" hypothesis.
Physics assumption: with the two C6 boards on a vertical axis, breathing motion
along the axis (lying flat) should give a stronger breath band than breathing
perpendicular to it (standing).

Outcome: the hypothesis was DISPROVEN (postural sway while standing dominates
the breath band) — see docs/EXPERIMENT_NOTES.md.

4 scenarios:
  1. standing_still   (standing, 60 s)
  2. sitting_still    (sitting, 60 s)
  3. tilted_recline   (half-reclined, 60 s)
  4. lying_flat       (lying flat, 60 s)

Usage: python record_posture_data.py --port COM5
"""
import argparse
import sys, os, time, csv, threading
from datetime import datetime
import numpy as np
import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kisum_radar_monitor import (
    SERIAL_PORT, SERIAL_BAUD, parse_csi_line,
    DopplerAnalyzer, DOPPLER_BANDS,
)

SCENARIOS = [
    ('standing_still',   60,
     'Start position: directly below the upper C6 (head ~40 cm below it, body vertical)',
     'Stand still for 60 s. Breathe normally (no breath-holding, no heavy breathing).\n'
     '     * Breathing motion is horizontal (chest moves front/back, perpendicular\n'
     '       to the vertical axis) -> expected: weak breath band'),

    ('sitting_still',    60,
     'Start position: chair on the vertical axis, seated (upper body on the axis)',
     'Sit still for 60 s. Breathe normally.\n'
     '     * Breathing motion is roughly horizontal -> expected: weak breath band\n'
     '       (similar to standing)'),

    ('tilted_recline',   60,
     'Start position: leaning against a headboard or chair back, body tilted\n'
     ' 30-45 deg, positioned on the vertical axis',
     'Stay half-reclined for 60 s. Breathe normally.\n'
     '     * Breathing motion is diagonal -> expected: medium breath band\n'
     '       (between standing and lying)'),

    ('lying_flat',       60,
     'Start position: bed directly below the vertical axis or < 50 cm to the side,\n'
     ' lying flat, chest facing the ceiling',
     '*** KEY TEST ***\n'
     '     Lie flat for 60 s, chest facing the ceiling. Breathe normally.\n'
     '     Breathing motion is vertical (chest up/down, along the axis)\n'
     '     -> expected: strong breath band\n'
     '     Head direction does not matter, as long as you face up.'),
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
    print(f'  → get ready, {seconds} s countdown ...')
    for i in range(int(seconds), 0, -1):
        print(f'    {i} ...', flush=True)
        time.sleep(1)
    print(f'  → recording started!', flush=True)


def ask_prep_seconds(default=3):
    while True:
        raw = input(f' >> Seconds needed to get into position? (Enter = default {default}, or a number like 15): ').strip()
        if raw == '':
            return default
        try:
            n = int(raw)
            if 1 <= n <= 60:
                return n
            print(f'    please enter an integer between 1 and 60')
        except ValueError:
            print(f'    please enter an integer (e.g. 3 or 15)')


def main():
    ap = argparse.ArgumentParser(description='Record 4 posture scenarios (CSI Doppler bands) to CSV.')
    ap.add_argument('--port', default=SERIAL_PORT,
                    help='serial port of the CSI receiver C6, e.g. COM5 (env CSI_PORT)')
    args = ap.parse_args()
    if not args.port:
        ap.error('no serial port given (use --port or set CSI_PORT)')
    port = args.port

    print('=' * 68)
    print(' Kisum posture data recorder')
    print('=' * 68)
    print()
    print(' Goal: test the "breathing direction -> posture" hypothesis.')
    print(' Hypothesis: on a vertical C6 axis, breath-band strength reflects the')
    print(' direction of chest motion.')
    print('       -> standing/sitting: horizontal breathing -> weak breath band')
    print('       -> reclined: in between')
    print('       -> lying flat: vertical breathing -> strong breath band')
    print()
    print(' Prepare 4 positions:')
    print('    1. a standing spot (between the two C6 boards)')
    print('    2. a chair (near the vertical axis)')
    print('    3. a reclining spot (chair back / headboard)')
    print('    4. a bed (lying flat, body crosswise)')
    print()
    print(' ! Keep all positions within 2 m of the line between the two C6 boards')
    print()

    os.makedirs('recordings', exist_ok=True)
    out_path = os.path.join('recordings',
        f'posture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    print(f' Output CSV: {out_path}')
    print()

    print(f' Opening {port} ...')
    try:
        c = Collector(out_path, port)
    except Exception as e:
        print(f' ERROR — cannot open serial port: {e}')
        print(f' Make sure kisum_radar_monitor.py is closed.')
        return 1

    print(f' Waiting 3 s for the Doppler pipeline to warm up ...')
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
        print(' All 4 scenarios recorded!')
        print('=' * 68)
        print(f' Total Doppler samples: {c.samples_written}')
        print(f' CSV saved to: {out_path}')
        print()
        print(' Next: compare the median breath band per scenario,')
        print('       in particular the lying_flat / standing_still ratio.')
    except KeyboardInterrupt:
        print('\n\n Interrupted. The data recorded so far has been saved.')
    finally:
        c.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
