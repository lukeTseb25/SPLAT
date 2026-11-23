#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mi_session_analysis.py

Analysis of motor-imagery recordings from the collection script: does mu
(8-13 Hz) or beta (13-20 Hz) power drop during imagery, and can it tell
the limbs apart?

Input: one or more recording CSVs (lsl_timestamp, CH1..CH8, marker).
Markers: 1 = right arm, 2 = left arm, 3 = right leg (cue / START),
4 = STOP. Unconnected channels (flat or railed at the amplifier limit) are
detected and left out automatically.

Electrode layout assumed (change LEFT/RIGHT/LATERAL/MEDIAL below if needed):
    CH3 left, near midline      CH4 right, near midline
    CH5 left, lateral           CH6 right, lateral

What it does, per recording
---------------------------
1. Causal 1-40 Hz band-pass + 60 Hz notch on the connected channels.
2. Moving-window bandpower over the whole recording (1 s Hann window,
   0.1 s step), in dB relative to the session median.
3. Cuts every trial from 6 s before to 6 s after its cue, and averages
   power over these periods (times are what the 1 s windows cover):
       rest       -5.0 to -3.5 s   (after the previous STOP)
       get_ready  -3.0 to  0.0 s   (Get Ready + instruction on screen)
       imagery     1.5 s to STOP   (skips the response to the cue)
       imagery_all 0.0 s to STOP
4. Reports, per session:
   A. Median % change in mu and beta during imagery vs each baseline, for
      each limb and channel (Wilcoxon test for a decrease).
   B. The FIXED lateralization test: mu, CH5 minus CH6, imagery vs rest.
      Rule: "right arm" if the left electrode (CH5) dropped more than the
      right one (CH6), else "left arm". Accuracy, AUC, binomial p.
      Run unchanged on every new session; that is what makes it a test.
   C. Imagery vs no imagery (beta, all connected channels): LDA, with both
      samples of a trial kept in the same fold. Against rest and Get Ready.
   D. Limb classification (LDA on mu + beta of the connected channels):
      3 limbs, right vs left arm. Cross-validated, shuffled-label p.
   E. Right leg vs arms: mean mu change, and (with all four channels)
      midline-minus-lateral.
5. Across sessions (when given more than one file): fixed test pooled,
   limb classification pooled, and train-on-others / test-on-one, using
   the channels all sessions share.
6. Figures: mu and beta across the trial cycle per session, and the
   left-minus-right electrode difference for right- vs left-arm trials.

Outputs (in --out-dir, default ./mi_analysis):
    analysis_report.txt     everything printed to the console
    changes_by_limb.csv     table A for every session
    trial_cycle.png         mu and beta over the trial, per session
    lateralization.png      left-minus-right electrode, right vs left arm

Usage
-----
    python mi_session_analysis.py rec1.csv rec2.csv ...
    python mi_session_analysis.py data/raw/*.csv --out-dir results --permutations 500
    python mi_session_analysis.py rec.csv --rf      # also run a random forest (slower)

Needs numpy, pandas, scipy, scikit-learn, matplotlib.
"""

import argparse
import io
import os
import sys
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, periodogram, sosfilt, sosfilt_zi, tf2sos
from scipy.stats import binomtest, mannwhitneyu, wilcoxon

warnings.filterwarnings("ignore")

# ============================================================================ settings
FS = 250.0
WIN_S, STEP_S = 1.0, 0.1
CANDIDATE_CHANNELS = ["CH3", "CH4", "CH5", "CH6"]
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 20.0)}
CLASSES = {"1": "right arm", "2": "left arm", "3": "right leg"}
STOP = "4"

LEFT, RIGHT = ["CH3", "CH5"], ["CH4", "CH6"]
MEDIAL, LATERAL = ["CH3", "CH4"], ["CH5", "CH6"]

# Periods relative to the cue (seconds the 1 s windows cover). "STOP" = cue-to-stop time.
PERIODS = {
    "rest": (-5.0, -3.5),
    "get_ready": (-3.0, 0.0),
    "imagery": (1.5, "STOP"),
    "imagery_all": (0.0, "STOP"),
}
BASELINES = ["rest", "get_ready"]
EPOCH = (-6.0, 6.0)           # cut around each cue for the time courses

# The fixed lateralization test (decided before seeing the Nov 9 session; keep it unchanged)
FIXED = {"band": "mu", "left": "CH5", "right": "CH6", "period": "imagery", "baseline": "rest"}

PRETTY = {"rest": "rest", "get_ready": "Get Ready", "imagery": "imagery 1.5 s-STOP",
          "imagery_all": "imagery 0 s-STOP"}


# ============================================================================ loading
def _norm_marker(v):
    s = "" if v is None else str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    except ValueError:
        return s


def load_recording(path):
    """Returns eeg (n, ch), channel names, markers aligned to EEG rows, connected-channel list."""
    df = pd.read_csv(path)
    marker_col = next((c for c in df.columns if c.lower().strip() == "marker"), df.columns[-1])
    present = [c for c in CANDIDATE_CHANNELS if c in df.columns]
    if not present:
        raise ValueError(f"{path}: none of {CANDIDATE_CHANNELS} found")
    eeg = df[present].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid_idx = np.flatnonzero(~np.isnan(eeg).any(axis=1))
    eeg = eeg[valid_idx]

    # Marker-only rows (blank EEG) go to the next EEG sample
    markers = [""] * len(valid_idx)
    for i, raw in enumerate(df[marker_col].tolist()):
        m = _norm_marker(raw)
        if m:
            j = np.searchsorted(valid_idx, i)
            if j < len(valid_idx) and not markers[j]:
                markers[j] = m

    # Unconnected = flat, or sitting at the amplifier rail (about +/-187500 uV on OpenBCI)
    connected = [c for k, c in enumerate(present)
                 if np.std(eeg[:, k]) > 1.0 and np.median(np.abs(eeg[:, k])) < 180000]
    keep = [present.index(c) for c in connected]
    return eeg[:, keep], connected, markers


# ============================================================================ signal
def prefilter(x, fs=FS, hp=1.0, lp=40.0, notch=60.0):
    sos = butter(4, [hp, lp], btype="bandpass", fs=fs, output="sos")
    if notch:
        b, a = iirnotch(notch, 30.0, fs)
        sos = np.vstack([sos, tf2sos(b, a)])
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    return sosfilt(sos, x, axis=0, zi=zi)[0]


def moving_bandpower(x, fs=FS, win_s=WIN_S, step_s=STEP_S):
    """Returns window centres (s) and power (n_windows, n_bands, n_channels)."""
    win, step = int(round(win_s * fs)), int(round(step_s * fs))
    freqs = np.fft.rfftfreq(win, 1 / fs)
    df = freqs[1] - freqs[0]
    lower_edges = {lo for lo, _ in BANDS.values()}
    masks = []
    for lo, hi in BANDS.values():
        upper = freqs < hi if hi in lower_edges else freqs <= hi   # 13 Hz counts as beta
        masks.append((freqs >= lo) & upper)
    starts = np.arange(0, x.shape[0] - win + 1, step)
    P = np.empty((len(starts), len(masks), x.shape[1]))
    for k, s in enumerate(starts):
        _, pxx = periodogram(x[s:s + win], fs=fs, window="hann", detrend="constant", axis=0)
        P[k] = [pxx[m].sum(axis=0) * df for m in masks]
    return (starts + win / 2) / fs, P


# ============================================================================ session
class Session:
    """One recording, cut into trials, with period averages ready to use."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        eeg, self.channels, markers = load_recording(path)
        t = np.arange(len(eeg)) / FS            # sample-index time (LSL stamps jitter)
        centres, P = moving_bandpower(prefilter(eeg))
        L = 10 * np.log10(P)
        L -= np.median(L[centres > 3], axis=0)  # dB relative to the session median

        cues = [(t[i], m) for i, m in enumerate(markers) if m in CLASSES]
        stops = np.array([t[i] for i, m in enumerate(markers) if m == STOP])
        durs = [stops[stops > tc][0] - tc for tc, _ in cues if (stops > tc).any()]
        self.imagery_s = float(np.median(durs)) if durs else 6.0

        self.rel = np.round(np.arange(EPOCH[0], EPOCH[1] + 1e-9, STEP_S), 3)
        labels, trials = [], []
        for tc, m in cues:
            if tc + EPOCH[0] - WIN_S / 2 < 3 or tc + EPOCH[1] + WIN_S / 2 > t[-1]:
                continue
            idx = np.clip(np.searchsorted(centres, tc + self.rel - 1e-6), 0, len(centres) - 1)
            trials.append(L[idx])
            labels.append(CLASSES[m])
        self.y = np.array(labels)
        self.T = np.stack(trials)               # (n_trials, n_times, n_bands, n_channels) dB
        self.band_names = list(BANDS)

    def period(self, name):
        """Mean dB over a period -> (n_trials, n_bands, n_channels)."""
        a, b = PERIODS[name]
        b = self.imagery_s if b == "STOP" else b
        lo, hi = a + WIN_S / 2, b - WIN_S / 2   # window centres fully inside the period
        k = (self.rel >= lo - 1e-9) & (self.rel <= hi + 1e-9)
        return self.T[:, k].mean(axis=1)

    def change(self, period, baseline):
        return self.period(period) - self.period(baseline)

    def ch(self, name):
        return self.channels.index(name)


# ============================================================================ stats helpers
pct = lambda db: 100 * (10 ** (np.asarray(db) / 10) - 1)


def p_decrease(v):
    v = np.asarray(v)
    if len(v) < 5 or np.allclose(v, 0):
        return np.nan
    return wilcoxon(v, alternative="less").pvalue


def auc(higher, lower):
    """P(a value from `higher` > a value from `lower`). 0.5 = no separation."""
    return mannwhitneyu(higher, lower).statistic / (len(higher) * len(lower))


def lateral_rule(idx, y):
    """idx = left electrode minus right electrode change. Negative -> 'right arm'."""
    k = np.isin(y, ["right arm", "left arm"])
    pred = np.where(idx[k] < 0, "right arm", "left arm")
    correct = int((pred == y[k]).sum())
    n = int(k.sum())
    return dict(correct=correct, n=n, acc=correct / n,
                p=binomtest(correct, n, 0.5, alternative="greater").pvalue,
                auc=auc(idx[k][y[k] == "left arm"], idx[k][y[k] == "right arm"]),
                right_ok=(pred[y[k] == "right arm"] == "right arm").mean(),
                left_ok=(pred[y[k] == "left arm"] == "left arm").mean())


def models(use_rf):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = {"LDA": (make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")), 20)}
    if use_rf:
        from sklearn.ensemble import RandomForestClassifier
        m["Random forest"] = (RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0), 4)
    return m


def cv_score(model, repeats, X, y, n_perm):
    from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                         cross_val_score, permutation_test_score)
    n_splits = max(2, min(5, min(np.sum(y == c) for c in set(y))))
    acc = cross_val_score(model, X, y, cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats,
                                                                  random_state=0)).mean()
    _, _, p = permutation_test_score(model, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
                                     n_permutations=n_perm, random_state=0, n_jobs=-1)
    return acc, p


def cv_paired(model, repeats, Xa, Xb, n_perm):
    """Xa vs Xb where row i of each comes from the same trial (kept in the same fold)."""
    from sklearn.model_selection import GroupKFold, cross_val_score, permutation_test_score
    X = np.vstack([Xa, Xb])
    y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    g = np.r_[np.arange(len(Xa)), np.arange(len(Xa))]
    acc = np.mean([cross_val_score(model, X, y, cv=GroupKFold(5),
                                   groups=np.random.default_rng(r).permutation(len(Xa))[g]).mean()
                   for r in range(repeats)])
    _, _, p = permutation_test_score(model, X, y, cv=GroupKFold(5), groups=g,
                                     n_permutations=n_perm, random_state=0, n_jobs=-1)
    return acc, p


fmt = lambda acc, p: f"{acc:4.0%} (p={p:.3f}){'*' if p < 0.05 else ' '}"


# ============================================================================ per-session report
def analyse_session(s, args, rows):
    print(f"\n{'=' * 78}\n{s.name}")
    print(f"  {len(s.y)} trials ({', '.join(f'{c} {np.sum(s.y == c)}' for c in CLASSES.values())}); "
          f"connected channels: {', '.join(s.channels)}; imagery {s.imagery_s:.1f} s")

    # ---- A. changes by limb
    for base in BASELINES:
        d = s.change("imagery", base)
        print(f"\n  A. Median % change, {PRETTY['imagery']} vs {PRETTY[base]}  (* = significant decrease, p<0.05)")
        print("     " + f"{'':10s}" + "".join(f"{b + ' ' + c:>12s}" for b in s.band_names for c in s.channels))
        for cl in CLASSES.values():
            k = s.y == cl
            cells = []
            for bi, b in enumerate(s.band_names):
                for ci, c in enumerate(s.channels):
                    v = d[k, bi, ci]
                    p = p_decrease(v)
                    cells.append(f"{pct(np.median(v)):+.0f}%{'*' if p < 0.05 else ' '}")
                    rows.append(dict(session=s.name, baseline=base, limb=cl, band=b, channel=c,
                                     median_change_pct=round(float(pct(np.median(v))), 1),
                                     share_of_trials_decreasing=round(float(np.mean(v < 0)), 2),
                                     p_decrease=round(float(p), 4)))
            print("     " + f"{cl:10s}" + "".join(f"{x:>12s}" for x in cells))

    # ---- B. fixed lateralization test
    f = FIXED
    if f["left"] in s.channels and f["right"] in s.channels:
        d = s.change(f["period"], f["baseline"])
        bi = s.band_names.index(f["band"])
        idx = d[:, bi, s.ch(f["left"])] - d[:, bi, s.ch(f["right"])]
        r = lateral_rule(idx, s.y)
        print(f"\n  B. FIXED TEST: {f['band']}, {f['left']} minus {f['right']}, {PRETTY[f['period']]} vs {PRETTY[f['baseline']]}")
        print(f"     accuracy {r['correct']}/{r['n']} = {r['acc']:.0%}  (p={r['p']:.3f}, chance 50%)  AUC {r['auc']:.2f}  "
              f"[right arm {r['right_ok']:.0%} correct, left arm {r['left_ok']:.0%}]")
        s.fixed_idx = idx
    else:
        print(f"\n  B. FIXED TEST skipped ({f['left']}/{f['right']} not both connected)")
        s.fixed_idx = None

    # Other left/right pairs and settings (exploratory)
    pairs = [(a, b) for a, b in zip(LEFT, RIGHT) if a in s.channels and b in s.channels]
    if pairs:
        print("\n     Other left-minus-right settings (exploratory; expect some to look good by chance):")
        for a, b in pairs:
            for bi, band in enumerate(s.band_names):
                cells = []
                for base in BASELINES:
                    d = s.change("imagery", base)
                    r = lateral_rule(d[:, bi, s.ch(a)] - d[:, bi, s.ch(b)], s.y)
                    cells.append(f"vs {PRETTY[base]}: {r['acc']:.0%} (AUC {r['auc']:.2f})")
                print(f"     {band:4s} {a}-{b}  " + "   ".join(cells))

    # ---- C. imagery vs no imagery (beta)
    bi = s.band_names.index("beta")
    print("\n  C. Imagery vs no imagery, beta in all connected channels (chance 50%)")
    for base in BASELINES:
        cells = [f"{name} {fmt(*cv_paired(m, min(rep, args.repeats), s.period('imagery')[:, bi], s.period(base)[:, bi], args.permutations))}"
                 for name, (m, rep) in models(args.rf).items()]
        print(f"     {PRETTY['imagery']} vs {PRETTY[base]:10s}: " + "   ".join(cells))

    # ---- D. limb classification
    X = s.period("imagery").reshape(len(s.y), -1)
    arms = np.isin(s.y, ["right arm", "left arm"])
    print("\n  D. Limb classification, mu + beta power during imagery")
    for name, (m, rep) in models(args.rf).items():
        print(f"     {name:14s} 3 limbs {fmt(*cv_score(m, min(rep, args.repeats), X, s.y, args.permutations))} (chance 33%)   "
              f"right vs left arm {fmt(*cv_score(m, min(rep, args.repeats), X[arms], s.y[arms], args.permutations))} (chance 50%)")

    # ---- E. leg vs arms
    print("\n  E. Right leg vs arms (mu, imagery vs rest)")
    d = s.change("imagery", "rest")[:, s.band_names.index("mu")]
    leg = s.y == "right leg"
    m_all = d.mean(axis=1)
    print(f"     mean of connected channels: leg {pct(np.median(m_all[leg])):+.0f}%, arms {pct(np.median(m_all[~leg])):+.0f}%, "
          f"AUC {auc(m_all[leg], m_all[~leg]):.2f} (p={mannwhitneyu(m_all[leg], m_all[~leg]).pvalue:.3f})")
    if all(c in s.channels for c in MEDIAL + LATERAL):
        ml = d[:, [s.ch(c) for c in MEDIAL]].mean(1) - d[:, [s.ch(c) for c in LATERAL]].mean(1)
        print(f"     midline minus lateral: AUC {auc(ml[~leg], ml[leg]):.2f} "
              f"(p={mannwhitneyu(ml[leg], ml[~leg], alternative='less').pvalue:.3f}; >0.5 = leg changes more at the midline)")


# ============================================================================ across sessions
def analyse_across(sessions, args):
    print(f"\n{'=' * 78}\nACROSS SESSIONS ({len(sessions)})")
    with_fixed = [s for s in sessions if s.fixed_idx is not None]
    if with_fixed:
        idx = np.concatenate([s.fixed_idx for s in with_fixed])
        y = np.concatenate([s.y for s in with_fixed])
        r = lateral_rule(idx, y)
        print(f"  Fixed test pooled ({len(with_fixed)} sessions): {r['correct']}/{r['n']} = {r['acc']:.0%} "
              f"(p={r['p']:.3f}), AUC {r['auc']:.2f}")

    common = [c for c in CANDIDATE_CHANNELS if all(c in s.channels for s in sessions)]
    if not common:
        print("  No channel is connected in every session; skipping pooled classification.")
        return
    feats = lambda s: s.period("imagery")[:, :, [s.ch(c) for c in common]].reshape(len(s.y), -1)
    Xs, ys = [feats(s) for s in sessions], [s.y for s in sessions]
    X, y = np.vstack(Xs), np.concatenate(ys)
    arms = np.isin(y, ["right arm", "left arm"])
    print(f"  Pooled limb classification on shared channels ({', '.join(common)}), {len(y)} trials:")
    for name, (m, rep) in models(args.rf).items():
        print(f"     {name:14s} 3 limbs {fmt(*cv_score(m, min(rep, args.repeats), X, y, args.permutations))}   "
              f"right vs left arm {fmt(*cv_score(m, min(rep, args.repeats), X[arms], y[arms], args.permutations))}")
    # Train on the other sessions, test on the held-out one. The model never sees the test
    # session, so this is the closest estimate of how it would do on a new recording.
    # The rest and Get Ready periods are controls: no imagery, so they should stay near 50%.
    m = models(False)["LDA"][0]

    def held_out(period, bands=(0, 1), shuffle_rng=None):
        Xp, yp = [], []
        for s in sessions:
            k = np.isin(s.y, ["right arm", "left arm"])
            Xp.append(s.period(period)[:, list(bands)][:, :, [s.ch(c) for c in common]].reshape(len(s.y), -1)[k])
            yp.append(shuffle_rng.permutation(s.y[k]) if shuffle_rng is not None else s.y[k])
        correct, per = 0, []
        for i in range(len(sessions)):
            tr = [j for j in range(len(sessions)) if j != i]
            m.fit(np.vstack([Xp[j] for j in tr]), np.concatenate([yp[j] for j in tr]))
            c = int((m.predict(Xp[i]) == yp[i]).sum())
            correct += c
            per.append(c / len(yp[i]))
        return correct, sum(len(v) for v in yp), per

    print(f"  Train on the other sessions, test on the held-out one (LDA, right vs left arm, "
          f"channels {', '.join(common)}, chance 50%):")
    for label, period, bands in [("mu + beta, imagery", "imagery", (0, 1)),
                                 ("mu only, imagery", "imagery", (0,)),
                                 ("beta only, imagery", "imagery", (1,)),
                                 ("CONTROL mu + beta, rest", "rest", (0, 1)),
                                 ("CONTROL mu + beta, Get Ready", "get_ready", (0, 1))]:
        c, n, per = held_out(period, bands)
        print(f"     {label:30s} {c}/{n} = {c / n:.0%}  (binomial p={binomtest(c, n, 0.5, alternative='greater').pvalue:.3f})"
              f"   per session: {' '.join(f'{v:.0%}' for v in per)}")
    c, n, _ = held_out("imagery")
    rng = np.random.default_rng(0)
    null = np.array([held_out("imagery", shuffle_rng=rng)[0] for _ in range(args.permutations)])
    print(f"     shuffled-label check (mu + beta, imagery): p = {(np.sum(null >= c) + 1) / (len(null) + 1):.3f}")
    print("     Sessions in order: " + ", ".join(s.name for s in sessions))


# ============================================================================ figures
def plot(sessions, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    ink, muted, surf, grid = "#1f1f1d", "#6b6a64", "#fcfcfb", "#e6e5e0"
    c1, c2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.labelcolor": ink})

    def frame(ax, s, title):
        ax.set_facecolor(surf)
        for a, b, txt in [(-5, -3.5, "rest"), (-3, 0, "get ready"), (0, s.imagery_s, "imagery")]:
            ax.axvspan(a, b, color=grid, alpha=0.5 if txt == "imagery" else 0.3, lw=0)
            ax.text((a + b) / 2, 0.97, txt, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color=muted)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_title(title, loc="left", color=ink, fontsize=10, fontweight="bold")
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(EPOCH)

    def line(ax, rel, trials, color, label):
        m = uniform_filter1d(trials.mean(0), 5)
        se = uniform_filter1d(trials.std(0) / np.sqrt(len(trials)), 5)
        ax.fill_between(rel, m - se, m + se, color=color, alpha=0.15, lw=0)
        ax.plot(rel, m, color=color, lw=2, label=label)

    n = len(sessions)
    cols = min(n, 3)
    rows_ = int(np.ceil(n / cols))

    # Figure 1: mu and beta over the trial (mean of trials and connected channels)
    fig, axs = plt.subplots(rows_, cols, figsize=(4.6 * cols, 3.3 * rows_), squeeze=False, facecolor=surf,
                            sharey=True)
    for ax, s in zip(axs.flat, sessions):
        frame(ax, s, s.name[:40])
        line(ax, s.rel, s.T[:, :, 0, :].mean(2), c1, "mu 8-13 Hz")
        line(ax, s.rel, s.T[:, :, 1, :].mean(2), c2, "beta 13-20 Hz")
    for ax in axs.flat[n:]:
        ax.axis("off")
    axs[0, 0].legend(loc="lower left", frameon=False, fontsize=8.5)
    for ax in axs[-1]:
        ax.set_xlabel("Seconds from cue")
    for ax in axs[:, 0]:
        ax.set_ylabel("dB vs session median")
    fig.suptitle("Mu and beta power across the trial (mean of trials and connected channels, ±1 SE)",
                 x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trial_cycle.png"), dpi=140, facecolor=surf)
    plt.close(fig)

    # Figure 2: left-minus-right lateral electrode, right- vs left-arm trials
    with_pair = [s for s in sessions if FIXED["left"] in s.channels and FIXED["right"] in s.channels]
    if not with_pair:
        return
    fig, axs = plt.subplots(len(with_pair), 2, figsize=(10, 2.8 * len(with_pair)), squeeze=False,
                            facecolor=surf, sharey=True)
    for r, s in enumerate(with_pair):
        a, b = s.ch(FIXED["left"]), s.ch(FIXED["right"])
        for c, (bi, band) in enumerate([(0, "mu"), (1, "beta")]):
            ax = axs[r, c]
            frame(ax, s, f"{s.name[:32]} · {band}")
            diff = s.T[:, :, bi, a] - s.T[:, :, bi, b]
            line(ax, s.rel, diff[s.y == "right arm"], c1, "right-arm trials")
            line(ax, s.rel, diff[s.y == "left arm"], c2, "left-arm trials")
    axs[0, 0].legend(loc="lower left", frameon=False, fontsize=8.5)
    for ax in axs[-1]:
        ax.set_xlabel("Seconds from cue")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"{FIXED['left']} − {FIXED['right']}, dB")
    fig.suptitle(f"{FIXED['left']} minus {FIXED['right']}: expected lower on right-arm trials during imagery",
                 x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lateralization.png"), dpi=140, facecolor=surf)
    plt.close(fig)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Mu/beta motor-imagery analysis of one or more recordings")
    ap.add_argument("csv", nargs="+", help="Recording CSV(s) from the collection script")
    ap.add_argument("--out-dir", default="mi_analysis")
    ap.add_argument("--permutations", type=int, default=200, help="Shuffles for each p-value")
    ap.add_argument("--repeats", type=int, default=20, help="Cross-validation repeats")
    ap.add_argument("--rf", action="store_true", help="Also run a random forest (slower)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, txt):
            sys.__stdout__.write(txt)
            sys.__stdout__.flush()
            buf.write(txt)
            return len(txt)

    rows, sessions = [], []
    with redirect_stdout(Tee()):
        print("Motor-imagery session analysis")
        print(f"Window {WIN_S:g} s, step {STEP_S:g} s; mu {BANDS['mu']} Hz, beta {BANDS['beta']} Hz; "
              f"{args.permutations} shuffles per p-value")
        for path in args.csv:
            try:
                s = Session(path)
            except Exception as e:
                print(f"\nSkipped {path}: {e}")
                continue
            sessions.append(s)
            analyse_session(s, args, rows)
        if len(sessions) > 1:
            analyse_across(sessions, args)

    if sessions:
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "changes_by_limb.csv"), index=False)
        plot(sessions, args.out_dir)
    with open(os.path.join(args.out_dir, "analysis_report.txt"), "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved report, table and figures to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
