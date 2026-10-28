# hybrid-body-bci

Software for the Hybrid Body Robotics EEG control interface.

Goal: read motor imagery off an 8-channel OpenBCI Cyton and turn it into a
discrete command (cursor left / right to start).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check it works with no hardware and no downloads:

```bash
python scripts/mi_analysis.py --simulate
python scripts/stream_cyton.py --synthetic
```

## What's here

| Path | What it does |
| --- | --- |
| `scripts/mi_analysis.py` | Decodes public motor-imagery data with CSP + LDA, and compares channel subsets so we know what 8 channels costs us |
| `scripts/stream_cyton.py` | Reads the Cyton over BrainFlow and converts to an MNE Raw. Runs on the synthetic board with no hardware |
| `src/loader.py` | The one function everything else calls. Loads a session folder (ours) or a public dataset and returns an MNE Raw with annotations attached |
| `src/contract.py` | Marker codes and channel montage, in one place so nobody hardcodes them |

## The data contract

A recording session is one folder:

```
data/sub-01_ses-01_task-mi_20261103/
    raw.csv          # straight from BrainFlow, no preprocessing
    events.csv       # trial, label, cue onset timestamp
    montage.json     # channel names in board order, reference, ground
    session.json     # subject, date, operator, impedances, board, notes
```

Marker codes are fixed. Do not change these:

| Code | Means |
| --- | --- |
| 1 | right hand cue |
| 2 | feet cue |
| 3 | rest |
| 9 | block boundary |

`src/loader.py` is the only thing that reads these files. If the format
changes, it changes there and nowhere else.

## Rules

Raw EEG does not go in the repo. `data/` is gitignored. Recordings live in
shared storage and the repo holds only code.

CSP gets fit inside the cross-validation fold, never before it. Fitting on all
the data first leaks labels and adds 10 to 20 points of accuracy that isn't
real. `mi_analysis.py` does this correctly; copy its pattern.

Report chance level next to every accuracy number.

## Groups

- Data: public datasets and our own recordings
- Electrode placement: montage, reference, artifact sources
- Model: raw signal to command
