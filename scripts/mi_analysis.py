"""
Motor imagery analysis: how much accuracy do we lose with 8 dry channels?

The question this answers: public motor-imagery datasets are recorded on 22-64
channel research rigs. We have an 8-channel OpenBCI Cyton. Before we commit to a
montage, decode the SAME trials with different channel subsets and compare.

Run it:
    pip install mne scikit-learn matplotlib numpy
    python mi_analysis.py                # real PhysioNet data, subjects 1-5
    python mi_analysis.py --subjects 1 2 3 4 5 6 7 8 9 10
    python mi_analysis.py --task feet    # hands vs feet instead of left vs right
    python mi_analysis.py --simulate     # no download, synthetic sanity check

Data is PhysioNet EEG Motor Movement/Imagery (eegmmidb), 109 subjects, 64 ch,
160 Hz. mne.datasets.eegbci downloads it for you, ~50 MB per subject.
Runs 4/8/12 are imagined left vs right fist. Runs 6/10/14 are imagined
hands vs feet.
"""

import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.decoding import CSP
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


# Channel sets to compare. Names are 10-10, as PhysioNet uses after
# eegbci.standardize(). Any channel missing from a recording is dropped with a
# warning rather than crashing.
CHANNEL_SETS = {
    "all64": None,  # everything the dataset has
    "motor8": ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"],
    "stock8": ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"],
    "c3czc4": ["C3", "Cz", "C4"],
}

# What each set represents, for the printed table.
CHANNEL_SET_NOTES = {
    "all64": "full research montage, the published benchmark",
    "motor8": "8 channels re-mapped onto the sensorimotor strip",
    "stock8": "Ultracortex Mark IV as it ships",
    "c3czc4": "the BNCI2014_004 montage, 3 channels",
}

TASK_RUNS = {"hands": [4, 8, 12], "feet": [6, 10, 14]}

FMIN, FMAX = 8.0, 30.0        # mu + beta
TMIN, TMAX = -1.0, 4.0        # epoch window around the cue
DECODE_TMIN, DECODE_TMAX = 0.5, 2.5  # analysis window, starts after the cue


def load_subject(subject, runs, data_path):
    """Download (if needed) and load one PhysioNet subject as a filtered Raw."""
    files = eegbci.load_data(subject, runs, path=data_path, update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    raw.set_eeg_reference("average", projection=False)
    raw.filter(FMIN, FMAX, fir_design="firwin", skip_by_annotation="edge")
    return raw


def make_epochs(raw):
    """Epoch around the two cue types. Returns epochs and integer labels."""
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    epochs = mne.Epochs(
        raw,
        events,
        event_id=dict(c1=1, c2=2),
        tmin=TMIN,
        tmax=TMAX,
        proj=False,
        baseline=None,
        preload=True,
        reject=dict(eeg=200e-6),
    )
    return epochs


def subset_channels(epochs, names):
    """Pick a channel subset, skipping any the recording does not have."""
    if names is None:
        return epochs.copy()
    have = [c for c in names if c in epochs.ch_names]
    missing = [c for c in names if c not in epochs.ch_names]
    if missing:
        print(f"      note: not in this recording, skipped: {missing}")
    if len(have) < 2:
        return None
    return epochs.copy().pick(have)


def score_subset(epochs, n_components=4, n_splits=5):
    """
    CSP + LDA, cross-validated. CSP is fit INSIDE each fold via the Pipeline,
    which is the whole point: fitting it on all the data first leaks labels and
    inflates accuracy by 10-20 points.

    Folds are not shuffled, so adjacent trials that share drift and electrode
    state do not land on both sides of a split.
    """
    X = epochs.get_data(tmin=DECODE_TMIN, tmax=DECODE_TMAX)
    y = epochs.events[:, -1]

    n_components = min(n_components, X.shape[1])
    clf = Pipeline([
        ("csp", CSP(n_components=n_components, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    chance = np.mean(y == np.bincount(y).argmax())
    return scores.mean(), scores.std(), chance, len(y)


def analyse(epochs, label=""):
    """Score every channel set on the same epochs."""
    out = {}
    for name, chans in CHANNEL_SETS.items():
        sub = subset_channels(epochs, chans)
        if sub is None:
            print(f"   {name}: too few channels present, skipped")
            continue
        mean, sd, chance, n = score_subset(sub)
        out[name] = dict(mean=mean, sd=sd, chance=chance, n_trials=n,
                         n_ch=len(sub.ch_names))
        print(f"   {name:8s} {len(sub.ch_names):3d} ch  "
              f"{mean*100:5.1f}% +/- {sd*100:4.1f}  "
              f"(chance {chance*100:.0f}%, {n} trials)  {label}")
    return out


def simulate_subject(n_trials=80, sfreq=160.0, seed=0):
    """
    Synthetic motor imagery for checking the pipeline runs and behaves.

    Builds a 64-channel recording of pink-ish noise, then suppresses mu/beta
    power on the side CONTRALATERAL to the imagined hand, which is the real
    effect (event-related desynchronization). Class 1 = left hand imagined, so
    the drop is at C4 and neighbours; class 2 = right hand, drop at C3.

    This is NOT a substitute for real data. It has no eye blinks, no muscle
    artifact, no between-subject variability, and the effect is cleaner than
    anything a scalp ever produces, so accuracy here will be optimistic. It
    exists to prove the code path works.
    """
    rng = np.random.default_rng(seed)
    montage = mne.channels.make_standard_montage("standard_1005")
    ch_names = [c for c in CHANNEL_SETS["motor8"] + CHANNEL_SETS["stock8"]
                + ["C1", "C2", "C5", "C6", "CPz", "Pz", "F3", "F4", "T7", "T8"]
                if c in montage.ch_names]
    ch_names = list(dict.fromkeys(ch_names))

    # ERD is strongest at C3/C4 and falls off across neighbours rather than
    # living on one electrode, which is why surrounding channels help CSP.
    left_weights = {"C3": 1.0, "C1": 0.7, "C5": 0.6, "FC3": 0.6, "CP3": 0.6}
    right_weights = {"C4": 1.0, "C2": 0.7, "C6": 0.6, "FC4": 0.6, "CP4": 0.6}

    trial_s = TMAX - TMIN
    n_samp = int(trial_s * sfreq)
    t = np.arange(n_samp) / sfreq
    cue_i = int(-TMIN * sfreq)

    data, labels = [], []
    for i in range(n_trials):
        cls = 1 if i % 2 == 0 else 2
        trial = rng.standard_normal((len(ch_names), n_samp)) * 22e-6

        # a mu rhythm on every channel, amplitude wandering trial to trial
        gain = rng.uniform(0.6, 1.4)
        for ci in range(len(ch_names)):
            ph = rng.uniform(0, 2 * np.pi)
            trial[ci] += gain * 14e-6 * np.sin(2 * np.pi * rng.uniform(9.5, 11.5) * t + ph)
            ph = rng.uniform(0, 2 * np.pi)
            trial[ci] += gain * 7e-6 * np.sin(2 * np.pi * rng.uniform(18, 24) * t + ph)

        # blink-ish drift, frontal only, uncorrelated with class
        for ci, name in enumerate(ch_names):
            if name in ("Fp1", "Fp2", "F3", "F4"):
                trial[ci] += rng.uniform(20e-6, 60e-6) * np.sin(
                    2 * np.pi * rng.uniform(0.5, 2.0) * t + rng.uniform(0, 6.3))

        # on ~20% of trials the subject's attention lapses and there is no ERD,
        # which is roughly what real motor imagery data looks like
        if rng.random() < 0.20:
            data.append(trial)
            labels.append(cls)
            continue

        weights = right_weights if cls == 1 else left_weights
        depth = rng.uniform(0.25, 0.45)   # fraction of power suppressed
        for ci, name in enumerate(ch_names):
            w = weights.get(name, 0.0)
            if w > 0:
                trial[ci, cue_i:] *= (1.0 - depth * w)

        data.append(trial)
        labels.append(cls)

    info = mne.create_info(ch_names, sfreq, "eeg")
    epochs = mne.EpochsArray(np.array(data), info, tmin=TMIN, verbose=False)
    epochs.set_montage(montage)
    epochs.events[:, -1] = np.array(labels)
    epochs.event_id = dict(c1=1, c2=2)
    return epochs


def plot_results(per_subject, task, out_png):
    """One grouped bar per channel set, one group per subject, chance marked."""
    sets = [s for s in CHANNEL_SETS if any(s in r for r in per_subject.values())]
    subs = sorted(per_subject)
    width = 0.8 / max(len(sets), 1)
    colors = ["#1f4e79", "#2e8b9a", "#d98c3f", "#8a8f94"]

    fig, ax = plt.subplots(figsize=(1.6 * len(subs) + 4, 5))
    for si, s in enumerate(sets):
        xs, ys = [], []
        for bi, sub in enumerate(subs):
            if s in per_subject[sub]:
                xs.append(bi + si * width - 0.4 + width / 2)
                ys.append(per_subject[sub][s]["mean"] * 100)
        ax.bar(xs, ys, width, label=f"{s} ({CHANNEL_SET_NOTES[s]})",
               color=colors[si % len(colors)])

    ax.axhline(50, color="#444", ls="--", lw=1)
    ax.text(len(subs) - 0.45, 51, "chance", fontsize=9, color="#444")
    ax.axhline(70, color="#b03030", ls=":", lw=1.2)
    ax.text(len(subs) - 0.45, 71, "usable BCI threshold", fontsize=9,
            color="#b03030")

    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([f"S{s}" for s in subs])
    ax.set_ylabel("cross-validated accuracy (%)")
    ax.set_ylim(35, 100)
    ax.set_title(f"CSP + LDA, {task}, by channel subset")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nplot written to {out_png}")


def summarise(per_subject):
    """Average each channel set across subjects and print the drop vs all64."""
    print("\n" + "=" * 68)
    print("MEAN ACROSS SUBJECTS")
    print("=" * 68)
    means = {}
    for s in CHANNEL_SETS:
        vals = [r[s]["mean"] for r in per_subject.values() if s in r]
        if vals:
            means[s] = float(np.mean(vals))

    base = means.get("all64")
    for s, m in means.items():
        delta = "" if base is None or s == "all64" else f"   {(m-base)*100:+5.1f} pts vs all64"
        print(f"  {s:8s} {m*100:5.1f}%   {CHANNEL_SET_NOTES[s]}{delta}")
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--task", choices=list(TASK_RUNS), default="hands")
    ap.add_argument("--data-path", default="./data")
    ap.add_argument("--out", default="channel_comparison.png")
    ap.add_argument("--simulate", action="store_true",
                    help="synthetic data, no download, for checking the code runs")
    args = ap.parse_args()

    per_subject = {}

    if args.simulate:
        print("SIMULATED DATA. Pipeline check only, these numbers mean nothing "
              "about real EEG.\n")
        for s in args.subjects:
            print(f"subject {s} (synthetic)")
            epochs = simulate_subject(seed=s)
            per_subject[s] = analyse(epochs)
        task_label = "synthetic left vs right hand imagery"
    else:
        runs = TASK_RUNS[args.task]
        task_label = ("imagined left vs right hand" if args.task == "hands"
                      else "imagined hands vs feet")
        for s in args.subjects:
            print(f"subject {s}, runs {runs}")
            try:
                raw = load_subject(s, runs, args.data_path)
                epochs = make_epochs(raw)
            except Exception as e:
                print(f"   failed: {type(e).__name__}: {e}")
                continue
            if len(epochs) < 20:
                print(f"   only {len(epochs)} clean epochs, skipping")
                continue
            per_subject[s] = analyse(epochs)

    if not per_subject:
        print("\nno subjects analysed.")
        return

    summarise(per_subject)
    plot_results(per_subject, task_label, args.out)


if __name__ == "__main__":
    main()
