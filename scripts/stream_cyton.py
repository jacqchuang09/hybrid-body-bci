"""
Read the Cyton over BrainFlow and hand back an MNE Raw.

Run it with no hardware at all:
    python scripts/stream_cyton.py --synthetic

Run it on the real board:
    python scripts/stream_cyton.py --port /dev/cu.usbserial-XXXX    # macOS
    python scripts/stream_cyton.py --port COM3                      # Windows
    python scripts/stream_cyton.py --port /dev/ttyUSB0              # Linux

On macOS the port must be /dev/cu.something, not /dev/tty.something. That one
detail accounts for most "it won't connect" reports.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def stream(port=None, seconds=10, synthetic=False):
    """
    Pull `seconds` of data off the board. Returns (data, board_id).

    data is the full BrainFlow matrix: rows are channels, columns are samples,
    and the rows interleave EEG with accelerometer, timestamp and marker rows.
    Find them by index, never hardcode.
    """
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

    params = BrainFlowInputParams()
    if synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD
    else:
        board_id = BoardIds.CYTON_BOARD
        if not port:
            raise SystemExit("real board needs --port (see the docstring)")
        params.serial_port = port

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    print(f"streaming {seconds}s ...")
    time.sleep(seconds)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    return data, board_id


def to_mne(data, board_id, ch_names=None):
    """
    BrainFlow matrix to MNE Raw.

    The divide by a million is not optional: BrainFlow returns microvolts and
    MNE expects volts.
    """
    import mne
    from brainflow.board_shim import BoardShim

    eeg_rows = BoardShim.get_eeg_channels(board_id)
    sfreq = BoardShim.get_sampling_rate(board_id)

    eeg = data[eeg_rows, :] / 1e6

    if ch_names is None:
        try:
            from src import contract
            ch_names = contract.CHANNEL_NAMES
        except Exception:
            ch_names = None
    if not ch_names or len(ch_names) != len(eeg_rows):
        # get_eeg_names returns OpenBCI's default assumption, which is wrong
        # the moment we re-map electrodes. Pass our real names instead.
        ch_names = BoardShim.get_eeg_names(board_id)[:len(eeg_rows)]

    info = mne.create_info(list(ch_names), sfreq, "eeg")
    raw = mne.io.RawArray(eeg, info, verbose=False)
    try:
        from src import contract
        raw.set_montage(contract.standard_montage("1020"), on_missing="ignore",
                        verbose=False)
    except Exception:
        pass
    return raw


def annotations_from_markers(raw, data, board_id):
    """
    Turn BrainFlow markers into MNE annotations.

    board.insert_marker(value) during recording writes into the marker row.
    Timing caveat: the marker lands on whichever sample the acquisition thread
    is holding, and nothing measures the gap between the cue appearing on
    screen and the stamp. Fine for motor imagery, where we average power over
    seconds. Not fine for ERP work.
    """
    import mne
    from brainflow.board_shim import BoardShim
    from src import contract

    ts = data[BoardShim.get_timestamp_channel(board_id), :]
    markers = data[BoardShim.get_marker_channel(board_id), :]
    idx = np.flatnonzero(markers)
    if len(idx) == 0:
        return raw

    raw.set_annotations(mne.Annotations(
        onset=ts[idx] - ts[0],
        duration=np.zeros(len(idx)),
        description=[contract.MARKERS.get(int(m), str(int(m)))
                     for m in markers[idx]]))
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--plot", default=None, help="save a PNG here")
    args = ap.parse_args()

    data, board_id = stream(args.port, args.seconds, args.synthetic)
    raw = to_mne(data, board_id)
    raw = annotations_from_markers(raw, data, board_id)

    print(raw)
    print("channels:", raw.ch_names)
    print("sfreq:", raw.info["sfreq"])
    print("duration: %.1f s" % (raw.n_times / raw.info["sfreq"]))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        fig = raw.plot(duration=5, n_channels=len(raw.ch_names), show=False,
                       scalings="auto")
        fig.savefig(args.plot, dpi=130)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
