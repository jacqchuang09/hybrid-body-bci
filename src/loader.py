"""
The single entry point for data, ours or public.

Everything downstream calls load_session() or load_public() and gets back an
MNE Raw with annotations already attached. If the file format changes, it
changes here and nowhere else.
"""

import json
import os

import numpy as np
import pandas as pd
import mne

from . import contract


def _apply_filters(raw, notch=True):
    if notch and raw.info["sfreq"] > 2 * contract.LINE_FREQ:
        freqs = [f for f in (contract.LINE_FREQ, 2 * contract.LINE_FREQ)
                 if f < raw.info["sfreq"] / 2]
        raw.notch_filter(freqs, verbose=False)
    raw.filter(contract.FMIN, contract.FMAX, fir_design="firwin", verbose=False)
    return raw


def load_session(folder, filtered=True):
    """
    Load one of our own recording sessions.

    folder holds raw.csv, events.csv, montage.json, session.json as described
    in the README. Returns an MNE Raw in volts with annotations set from
    events.csv.
    """
    with open(os.path.join(folder, "montage.json")) as f:
        montage = json.load(f)
    with open(os.path.join(folder, "session.json")) as f:
        meta = json.load(f)

    ch_names = montage["channel_names"]
    sfreq = float(meta.get("sfreq", contract.SFREQ))

    # BrainFlow CSV: rows are samples, columns are the board's channel layout.
    # eeg_columns says which columns are EEG, so we don't guess indices.
    table = pd.read_csv(os.path.join(folder, "raw.csv"))
    eeg_cols = montage["eeg_columns"]
    data = table[eeg_cols].to_numpy().T

    # BrainFlow reports microvolts, MNE wants volts. Skipping this puts every
    # threshold off by a factor of a million.
    data = data / 1e6

    info = mne.create_info(ch_names, sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    try:
        raw.set_montage(contract.standard_montage("1020"), on_missing="warn",
                        verbose=False)
    except Exception:
        pass

    ev_path = os.path.join(folder, "events.csv")
    if os.path.exists(ev_path):
        events = pd.read_csv(ev_path)
        t0 = float(table[montage.get("timestamp_column", "timestamp")].iloc[0]) \
            if montage.get("timestamp_column") in table else 0.0
        onsets = events["cue_onset"].to_numpy(dtype=float) - t0
        descs = [contract.MARKERS.get(int(c), str(int(c)))
                 for c in events["label"]]
        raw.set_annotations(mne.Annotations(
            onset=onsets, duration=np.zeros(len(onsets)), description=descs))

    raw.info["description"] = json.dumps(
        {"session": os.path.basename(folder.rstrip("/")),
         "montage_version": montage.get("version", "unknown")})

    return _apply_filters(raw) if filtered else raw


def load_public(dataset="BNCI2014_004", subject=1, filtered=True):
    """
    Load a public motor-imagery dataset through MOABB.

    BNCI2014_004 is three channels at C3, Cz, C4, which is the closest public
    recording to our hardware. Others worth trying: BNCI2014_001 (22 ch, the
    BCI Competition IV 2a benchmark), Cho2017 (52 subjects), PhysionetMI.

    Returns a dict of session name -> run name -> Raw, the MOABB shape.
    """
    import moabb.datasets as mds

    ds = getattr(mds, dataset)()
    data = ds.get_data(subjects=[subject])[subject]

    if filtered:
        for sess in data.values():
            for run_name, raw in sess.items():
                sess[run_name] = _apply_filters(raw.load_data())
    return data


def epochs_from_raw(raw, classes=None, tmin=-1.0, tmax=4.0, reject_uv=150):
    """
    Cut a Raw into labelled epochs around the cue.

    Window and classes come from contract.py so the offline and live paths
    agree. reject_uv drops epochs with a peak-to-peak swing above that, which
    is usually a blink or a movement.
    """
    classes = classes or contract.TASK_CLASSES
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {k: v for k, v in event_id.items() if k in classes}
    if not keep:
        raise ValueError(
            f"none of {classes} found in annotations; present: {list(event_id)}")

    return mne.Epochs(
        raw, events, event_id=keep, tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
        reject=dict(eeg=reject_uv * 1e-6) if reject_uv else None,
        verbose=False)


def ring_buffer_windows(raw, window_s=2.0, step_s=0.25):
    """
    Yield the sliding windows the live decoder will see.

    This is the answer to 'do we need past EEG or just current': neither. Keep
    the last window_s seconds, classify, slide forward by step_s, discard the
    rest. Use this offline too so the live and offline paths can't diverge.
    """
    sfreq = raw.info["sfreq"]
    n_win = int(window_s * sfreq)
    n_step = int(step_s * sfreq)
    data = raw.get_data()
    for start in range(0, data.shape[1] - n_win + 1, n_step):
        yield start / sfreq, data[:, start:start + n_win]
