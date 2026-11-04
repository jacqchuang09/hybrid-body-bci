"""
Everything about our data format lives here. One place, so no file goes
hardcoding a marker code or a channel name.
"""

# Fixed. Changing a code invalidates every recording made before the change,
# so if you need a new one, add it, don't renumber.
MARKERS = {
    1: "right_hand",
    2: "feet",
    3: "rest",
    9: "block_boundary",
}

LABEL_TO_CODE = {v: k for k, v in MARKERS.items()}

# The two classes Phase 1 decodes. Hand versus feet separates better than
# left versus right hand, which matters at 8 channels.
TASK_CLASSES = ["right_hand", "feet"]

# Our 8 channels, in the order the Cyton reports them. Update this when the
# montage is settled with hardware, and bump MONTAGE_VERSION.
MONTAGE_VERSION = "draft-1"
CHANNEL_NAMES = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]

REFERENCE = "linked_mastoid"   # ear clips to SRB2
GROUND = "ear_clip_bias"

SFREQ = 250.0                  # Cyton default

# Band for motor imagery: mu + beta.
FMIN, FMAX = 8.0, 30.0
LINE_FREQ = 60.0               # US

# Trial timing, Graz 001-2015 shape. Seconds relative to cue onset.
CUE_AT = 2.0
IMAGERY_START, IMAGERY_END = 0.0, 4.0
DECODE_WINDOW = (0.5, 3.5)     # starts after the cue, avoids the evoked response
ITI_RANGE = (2.0, 3.0)         # random, so subjects can't anticipate

TRIALS_PER_CLASS_PER_RUN = 40
RUNS_PER_SESSION = 5


def standard_montage(system="1020"):
    """
    MNE's template electrode positions for the 10-20 ("1020") or 10-05
    ("1005") system.

    MNE 1.13 renamed these from standard_* to colin27_* and drops the old
    names in 1.14. Older MNE only has the old names, so try the new one first.
    """
    import mne

    if f"colin27_{system}" in mne.channels.get_builtin_montages():
        return mne.channels.make_standard_montage(f"colin27_{system}")
    return mne.channels.make_standard_montage(f"standard_{system}")


def describe():
    """Print the contract, for sanity-checking at the start of a session."""
    print(f"montage {MONTAGE_VERSION}: {CHANNEL_NAMES}")
    print(f"reference {REFERENCE}, ground {GROUND}, {SFREQ} Hz")
    print(f"band {FMIN}-{FMAX} Hz, notch {LINE_FREQ} Hz")
    print(f"classes {TASK_CLASSES}, decode window {DECODE_WINDOW} s")
    print("markers " + ", ".join(f"{k}={v}" for k, v in MARKERS.items()))


if __name__ == "__main__":
    describe()
