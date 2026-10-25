#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mi_epoch_bandpower.py

Epoch a motor-imagery recording at each cue marker and compute mu (8-13 Hz)
and beta (13-20 Hz) bandpower for channels 3-6 using a moving window.
The output is a feature table for a dot/dash classifier; no detection or
classification is done here.

Input: CSV with CH3, CH4, CH5, CH6 and a marker column (an optional
lsl_timestamp column is used if present). Raw recordings from the collection
script (CH1..CH8 + marker) also work; CH3..CH6 are picked by name.

Epochs:
    imagery  : from each cue marker (1/2/3) to the next 4 marker.
               A cue with no 4 before the next cue is skipped.
    baseline : [-5.5, -3.5] s relative to the cue by default (--baseline);
               the rest period before "Get Ready" in the collection script.

Inside each epoch a window (default 1.0 s, step 0.1 s) slides across the
data. For each window: Hann periodogram -> power summed over the band.

Outputs (next to the input file, or --out-dir):
    <name>_epoch_features.csv  one row per epoch:
        epoch, marker, label, cue_time, stop_time, duration_s, n_windows,
        {band}_{ch}_power        mean bandpower over the imagery windows (µV²)
        {band}_{ch}_baseline     mean bandpower over the baseline windows (µV²)
        {band}_{ch}_change_pct   (power - baseline) / baseline * 100
                                 (negative = decrease)
    <name>_epoch_windows.csv   one row per window (for time courses)
    <name>_epochs.npz          arrays for training:
        power      (n_epochs, max_windows, n_bands, n_channels), NaN-padded
                   because epochs can differ in length
        n_windows  (n_epochs,) real windows in each epoch
        baseline   (n_epochs, n_bands, n_channels)
        markers, labels, cue_times, stop_times,
        window_times (window end, s after the cue),
        bands, channels

Usage:
    python mi_epoch_bandpower.py data/raw/MI_EEG_20261025_120000.csv
    python mi_epoch_bandpower.py rec.csv --window 0.5 --step 0.05
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, periodogram, sosfilt, sosfilt_zi, tf2sos

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("epoch_bandpower")

CHANNELS = ["CH3", "CH4", "CH5", "CH6"]
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 20.0)}
CUE_MARKERS = {"1": "right arm", "2": "left arm", "3": "right leg"}
STOP_MARKER = "4"


# ----------------------------------------------------------------------------- loading
def _norm_marker(v) -> str:
    s = "" if v is None else str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    except ValueError:
        return s


def load_recording(path, fs):
    """Returns timestamps (n,), eeg (n, 4), markers (list of str, '' = none)."""
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    marker_col = lower.get("marker", df.columns[-1])
    ts_col = next((c for lc, c in lower.items() if "time" in lc), None)

    if all(c in df.columns for c in CHANNELS):
        ch_cols = CHANNELS
    else:
        others = [c for c in df.columns if c not in (marker_col, ts_col)]
        ch_cols = others[:4]
        log.warning(f"CH3..CH6 not found by name; using columns {ch_cols}")

    eeg = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid_idx = np.flatnonzero(~np.isnan(eeg).any(axis=1))
    if valid_idx.size == 0:
        raise ValueError("No complete EEG rows")

    # Marker-only rows (blank EEG) -> attach to the next EEG sample
    markers = [""] * valid_idx.size
    for i, raw in enumerate(df[marker_col].tolist()):
        m = _norm_marker(raw)
        if not m:
            continue
        j = np.searchsorted(valid_idx, i)
        if j < valid_idx.size and not markers[j]:
            markers[j] = m

    ts = None
    if ts_col is not None:
        ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy()[valid_idx]
        if np.isnan(ts).any() or np.any(np.diff(ts) <= 0):
            log.warning("Bad timestamps; using sample index / fs")
            ts = None
    if ts is None:
        ts = np.arange(valid_idx.size) / fs

    log.info(f"{valid_idx.size} samples ({valid_idx.size / fs:.1f} s), "
             f"{sum(1 for m in markers if m)} markers")
    return ts, eeg[valid_idx], markers


# ----------------------------------------------------------------------------- signal
def prefilter(eeg, fs, hp=1.0, lp=40.0, notch=60.0):
    """Causal band-pass + notch (same filter you'd run online)."""
    sos = butter(4, [hp, lp], btype="bandpass", fs=fs, output="sos")
    if notch:
        b, a = iirnotch(notch, 30.0, fs)
        sos = np.vstack([sos, tf2sos(b, a)])
    zi = sosfilt_zi(sos)[:, :, None] * eeg[0][None, None, :]
    return sosfilt(sos, eeg, axis=0, zi=zi)[0]


def band_masks(freqs):
    lower_edges = {lo for lo, _ in BANDS.values()}
    masks = []
    for lo, hi in BANDS.values():
        upper = freqs < hi if hi in lower_edges else freqs <= hi  # 13 Hz -> beta
        m = (freqs >= lo) & upper
        if not m.any():
            raise ValueError(f"No FFT bins in {lo}-{hi} Hz; use a longer window")
        masks.append(m)
    return masks


def moving_bandpower(x, fs, win, step):
    """
    x: (n_samples, n_ch) -> (window end offsets in samples, power (n_win, n_bands, n_ch))
    """
    freqs = np.fft.rfftfreq(win, 1.0 / fs)
    df = freqs[1] - freqs[0]
    masks = band_masks(freqs)
    starts = np.arange(0, x.shape[0] - win + 1, step)
    out = np.empty((len(starts), len(masks), x.shape[1]))
    for k, s in enumerate(starts):
        _, pxx = periodogram(x[s:s + win], fs=fs, window="hann", detrend="constant", axis=0)
        out[k] = [pxx[m].sum(axis=0) * df for m in masks]
    return starts + win, out


def pair_epochs(ts, markers):
    """[(cue_time, stop_time, cue_marker)] : each cue 1/2/3 closed by the next 4."""
    epochs, open_cue = [], None
    for i, m in enumerate(markers):
        if m in CUE_MARKERS:
            if open_cue is not None:
                log.warning(f"Cue at t={open_cue[0]:.2f} has no 4 marker before the next cue; skipped")
            open_cue = (ts[i], m)
        elif m == STOP_MARKER:
            if open_cue is None:
                log.warning(f"4 marker at t={ts[i]:.2f} without a preceding cue; ignored")
                continue
            epochs.append((open_cue[0], ts[i], open_cue[1]))
            open_cue = None
    if open_cue is not None:
        log.warning(f"Last cue at t={open_cue[0]:.2f} has no 4 marker (recording ended); skipped")
    return epochs


def segment(eeg_f, ts, t0, t1):
    i0, i1 = np.searchsorted(ts, t0), np.searchsorted(ts, t1)
    return eeg_f[i0:i1] if i0 < i1 else None


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Per-epoch mu/beta bandpower (moving window)")
    p.add_argument("csv")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--fs", type=float, default=250.0)
    p.add_argument("--baseline", type=float, nargs=2, default=[-5.5, -3.5], metavar=("START", "END"),
                   help="Baseline relative to cue (s)")
    p.add_argument("--window", type=float, default=1.0, help="Moving window length (s)")
    p.add_argument("--step", type=float, default=0.1, help="Moving window step (s)")
    p.add_argument("--notch", type=float, default=60.0, help="Line-noise notch Hz (0 = off)")
    a = p.parse_args()

    fs = a.fs
    win, step = int(round(a.window * fs)), max(1, int(round(a.step * fs)))
    ts, eeg, markers = load_recording(a.csv, fs)
    eeg_f = prefilter(eeg, fs, notch=a.notch or None)

    epochs = pair_epochs(ts, markers)
    if not epochs:
        raise SystemExit("No complete epochs (cue 1/2/3 followed by a 4) found")

    band_names = list(BANDS)
    rows, win_rows, P_all, B_all, kept = [], [], [], [], []

    for n, (tc, tstop, m) in enumerate(epochs, 1):
        ep = segment(eeg_f, ts, tc, tstop)
        bl = segment(eeg_f, ts, tc + a.baseline[0], tc + a.baseline[1])
        if ep is None or ep.shape[0] < win:
            log.warning(f"Epoch {n} at t={tc:.2f} is shorter than one window; skipped")
            continue
        ends, P = moving_bandpower(ep, fs, win, step)
        rel_times = ends / fs  # window end, seconds after the cue

        if bl is not None and bl.shape[0] >= win:
            B = moving_bandpower(bl, fs, win, step)[1].mean(axis=0)
        else:
            log.warning(f"Epoch {n}: baseline outside recording; change_pct = NaN")
            B = np.full(P.shape[1:], np.nan)

        Pm = P.mean(axis=0)
        change = (Pm - B) / B * 100.0

        row = {"epoch": n, "marker": m, "label": CUE_MARKERS[m], "cue_time": tc,
               "stop_time": tstop, "duration_s": tstop - tc, "n_windows": P.shape[0]}
        for b, band in enumerate(band_names):
            for c, ch in enumerate(CHANNELS):
                row[f"{band}_{ch}_power"] = Pm[b, c]
                row[f"{band}_{ch}_baseline"] = B[b, c]
                row[f"{band}_{ch}_change_pct"] = change[b, c]
        rows.append(row)

        for k in range(P.shape[0]):
            wr = {"epoch": n, "marker": m, "t_rel": rel_times[k]}
            for b, band in enumerate(band_names):
                for c, ch in enumerate(CHANNELS):
                    wr[f"{band}_{ch}_power"] = P[k, b, c]
            win_rows.append(wr)

        P_all.append(P)
        B_all.append(B)
        kept.append((m, CUE_MARKERS[m], tc, tstop))

    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.csv))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.csv))[0]
    feat_path = os.path.join(out_dir, f"{stem}_epoch_features.csv")
    win_path = os.path.join(out_dir, f"{stem}_epoch_windows.csv")
    npz_path = os.path.join(out_dir, f"{stem}_epochs.npz")

    feats = pd.DataFrame(rows)
    feats.to_csv(feat_path, index=False)
    pd.DataFrame(win_rows).to_csv(win_path, index=False)
    # Epochs differ in length -> pad the window axis with NaN
    n_win = np.array([P.shape[0] for P in P_all])
    power = np.full((len(P_all), n_win.max()) + P_all[0].shape[1:], np.nan)
    for i, P in enumerate(P_all):
        power[i, :P.shape[0]] = P
    window_times = (win + np.arange(n_win.max()) * step) / fs
    np.savez(npz_path, power=power, n_windows=n_win, baseline=np.stack(B_all),
             markers=np.array([k[0] for k in kept]), labels=np.array([k[1] for k in kept]),
             cue_times=np.array([k[2] for k in kept]), stop_times=np.array([k[3] for k in kept]),
             window_times=window_times, bands=np.array(band_names), channels=np.array(CHANNELS))

    dur = feats["duration_s"]
    log.info(f"{len(rows)} epochs, {dur.min():.2f}-{dur.max():.2f} s long, "
             f"{n_win.min()}-{n_win.max()} windows each ({a.window:g} s window, {a.step:g} s step)")
    for path in (feat_path, win_path, npz_path):
        log.info(f"Wrote {path}")

    cols = [f"{b}_{c}_change_pct" for b in band_names for c in CHANNELS]
    print("\nMean % change from baseline by class (negative = decrease):")
    summary = feats.groupby("label")[cols].mean().round(1)
    summary.columns = [c.replace("_change_pct", "") for c in summary.columns]
    print(summary.to_string())


if __name__ == "__main__":
    main()