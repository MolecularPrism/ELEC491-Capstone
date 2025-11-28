# evaluate_event_tflite.py
# Event-level (per-trial) evaluation using a quantized TFLite model (INT8/UINT8/FP32).
# Detection rule (fall trial):
#   detected=True if exists at least one window predicted as fall_class and its end_frame satisfies:
#       onset_frame - pre_onset_sec*fs <= end_frame <= impact_frame
# Detection rule (ADL trial):
#   detected=True if ANY window in the whole trial is predicted as fall_class (false positive).
#
# IO conventions:
# - CSV signal columns: AccX, AccY, AccZ, GyrX, GyrY, GyrZ (case-insensitive prefix match).
# - Normalization uses train data stats: data_root/train/X.npy (N,T,C), mean/std over axes (0,1).
# - Each window (win_len,6) -> normalize -> transpose -> (6,win_len) -> expand -> (6,win_len,1)
# - Then quantize to TFLite input dtype if needed (int8/uint8).

from __future__ import annotations

import os
import re
import json
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


# =========================
# Label parsing
# =========================
def parse_task_id_from_cell(cell: str) -> Optional[int]:
    """Extract task_id from a cell like 'F01 (20)' -> 20."""
    if pd.isna(cell):
        return None
    m = re.search(r"\((\d+)\)", str(cell))
    return int(m.group(1)) if m else None


def load_label_table(xlsx_path: str) -> pd.DataFrame:
    """Parse label excel into standardized columns: [task_id, trial_id, onset_frame, impact_frame]."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    col_task = next((c for c in df.columns if "Task Code" in c or "Task" in c), None)
    col_trial = next((c for c in df.columns if "Trial" in c and "ID" in c), None)
    col_onset = next((c for c in df.columns if "onset" in c.lower()), None)
    col_impact = next((c for c in df.columns if "impact" in c.lower()), None)
    if not all([col_task, col_trial, col_onset, col_impact]):
        raise ValueError(f"Cannot identify required columns in {xlsx_path}. Columns: {df.columns.tolist()}")

    # Forward fill Task Code down the column
    df[col_task] = df[col_task].ffill()
    # Extract task_id from 'F01 (20)'
    df["task_id"] = df[col_task].apply(parse_task_id_from_cell)

    out = df[["task_id", col_trial, col_onset, col_impact]].copy()
    out.columns = ["task_id", "trial_id", "onset_frame", "impact_frame"]

    out = out.dropna(subset=["task_id", "trial_id", "onset_frame", "impact_frame"])
    out["task_id"] = out["task_id"].astype(int)
    out["trial_id"] = out["trial_id"].astype(int)
    out["onset_frame"] = out["onset_frame"].astype(int)
    out["impact_frame"] = out["impact_frame"].astype(int)
    return out


def parse_labels(excel_path: str, subject_prefix: str) -> List[Dict]:
    """Build label entries: task_id, trial_id, onset_frame, impact_frame, csv_name."""
    df = load_label_table(excel_path)
    entries: List[Dict] = []
    for _, row in df.iterrows():
        task_id = int(row["task_id"])
        trial_id = int(row["trial_id"])
        onset = int(row["onset_frame"])
        impact = int(row["impact_frame"])
        csv_name = f"{subject_prefix}T{task_id:02d}R{trial_id:02d}.csv"
        entries.append(
            dict(
                task_id=task_id,
                trial_id=trial_id,
                onset_frame=onset,
                impact_frame=impact,
                csv_name=csv_name,
            )
        )
    return entries


# =========================
# Normalization stats (match training)
# =========================
def compute_norm_stats_from_train(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std from data_root/train/X.npy with shape (N,T,C)."""
    train_x_path = os.path.join(data_root, "train", "X.npy")
    if not os.path.exists(train_x_path):
        raise FileNotFoundError(f"train X.npy not found: {train_x_path}")

    train_X = np.load(train_x_path)
    if train_X.ndim != 3:
        raise ValueError(f"train X.npy has invalid shape: {train_X.shape}, expected (N,T,C)")

    mean = train_X.mean(axis=(0, 1)).astype(np.float32)  # (C,)
    std = train_X.std(axis=(0, 1)).astype(np.float32)    # (C,)
    std = np.where(std == 0, 1.0, std).astype(np.float32)
    return mean, std


# =========================
# Sensor CSV parsing
# =========================
def load_signal_from_csv(csv_path: str) -> np.ndarray:
    """Load 6-axis signal from one trial CSV. Return shape: (N_frames, 6)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    def pick(prefix: str) -> str:
        cands = [c for c in df.columns if str(c).lower().startswith(prefix.lower())]
        if not cands:
            raise KeyError(f"Column starting with '{prefix}' not found in {csv_path}")
        return cands[0]

    accx = pick("AccX")
    accy = pick("AccY")
    accz = pick("AccZ")
    gyrx = pick("GyrX")
    gyry = pick("GyrY")
    gyrz = pick("GyrZ")

    return df[[accx, accy, accz, gyrx, gyry, gyrz]].values.astype(np.float32)


def parse_sensor_filename(fname: str) -> Optional[Dict]:
    """Parse S06T20R01.csv => subj=6, task_id=20, trial_id=1."""
    m = re.match(r"S(?P<subj>\d+)T(?P<task>\d{2})R(?P<rep>\d{2})\.csv$", fname, re.IGNORECASE)
    if not m:
        return None
    return {
        "subj": int(m.group("subj")),
        "task_id": int(m.group("task")),
        "trial_id": int(m.group("rep")),
    }


# =========================
# Sliding windows
# =========================
def sliding_windows_until_impact(
    signal: np.ndarray,
    impact_frame: int,
    win_len_frames: int,
    step: int = 10,
) -> List[Tuple[np.ndarray, int]]:
    """Generate windows ending from win_len_frames to impact_frame (inclusive), with step; end_frame is 1-based."""
    N = int(signal.shape[0])
    impact = min(int(impact_frame), N)
    windows: List[Tuple[np.ndarray, int]] = []

    for end_frame in range(win_len_frames, impact + 1, step):
        start_idx = end_frame - win_len_frames
        end_idx = end_frame
        w = signal[start_idx:end_idx, :]
        if w.shape[0] != win_len_frames:
            continue
        windows.append((w, end_frame))
    return windows


def sliding_windows_full(
    signal: np.ndarray,
    win_len_frames: int,
    step: int = 10,
) -> List[Tuple[np.ndarray, int]]:
    """Generate windows over the entire trial; end_frame is 1-based."""
    N = int(signal.shape[0])
    windows: List[Tuple[np.ndarray, int]] = []

    for end_frame in range(win_len_frames, N + 1, step):
        start_idx = end_frame - win_len_frames
        end_idx = end_frame
        w = signal[start_idx:end_idx, :]
        if w.shape[0] != win_len_frames:
            continue
        windows.append((w, end_frame))
    return windows


# =========================
# TFLite inference helpers
# =========================
def make_interpreter(tflite_path: str) -> tf.lite.Interpreter:
    """Create and allocate a TFLite interpreter."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def get_io_details(interpreter: tf.lite.Interpreter) -> Tuple[Dict, Dict]:
    """Return (input_detail, output_detail) for single-input single-output models."""
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    if len(in_details) != 1 or len(out_details) != 1:
        raise ValueError(f"Expected single input/output, got {len(in_details)} inputs and {len(out_details)} outputs.")
    return in_details[0], out_details[0]


def preprocess_window_float(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Match training preprocessing and return float32 NHWC sample:
      window: (win_len,6) -> normalize -> transpose -> (6,win_len) -> expand -> (6,win_len,1)
    """
    std_safe = np.where(std == 0, 1.0, std).astype(np.float32)
    x = window.astype(np.float32)
    x = (x - mean.astype(np.float32)) / std_safe
    x = x.T  # (6,win_len)
    x = np.expand_dims(x, axis=-1)  # (6,win_len,1)
    return x.astype(np.float32)


def quantize_to_input(x_float: np.ndarray, in_detail: Dict) -> np.ndarray:
    """
    Quantize float input to match TFLite input dtype (int8/uint8) if needed.
    Returns array shaped (1,H,W,C) matching interpreter input.
    """
    # Ensure batch dimension exists
    if x_float.ndim == 3:
        x_float = np.expand_dims(x_float, axis=0)  # (1,H,W,C)

    dtype = in_detail["dtype"]
    scale, zero_point = in_detail.get("quantization", (0.0, 0))

    if dtype == np.float32:
        return x_float.astype(np.float32)

    if scale is None or scale == 0.0:
        raise ValueError("Input tensor is quantized but scale is 0. Cannot quantize input properly.")

    q = np.round(x_float / scale + zero_point)

    if dtype == np.int8:
        q = np.clip(q, -128, 127).astype(np.int8)
        return q
    if dtype == np.uint8:
        q = np.clip(q, 0, 255).astype(np.uint8)
        return q

    raise ValueError(f"Unsupported input dtype: {dtype}")


def argmax_from_output(y: np.ndarray) -> int:
    """Compute argmax class index from output tensor (supports quantized or float)."""
    y = np.array(y)
    if y.ndim == 2:
        y = y[0]
    return int(np.argmax(y))


def predict_class_tflite(
    interpreter: tf.lite.Interpreter,
    in_detail: Dict,
    out_detail: Dict,
    x_float_hwc: np.ndarray,
) -> int:
    """Run one inference and return predicted class index."""
    x_in = quantize_to_input(x_float_hwc, in_detail)

    interpreter.set_tensor(in_detail["index"], x_in)
    interpreter.invoke()
    y = interpreter.get_tensor(out_detail["index"])

    # For argmax, dequantization is unnecessary (linear transform preserves ordering),
    # as long as dtype/shape are correct.
    return argmax_from_output(y)


# =========================
# Event-level evaluation per trial
# =========================
def evaluate_one_fall_trial_tflite(
    interpreter: tf.lite.Interpreter,
    in_detail: Dict,
    out_detail: Dict,
    signal: np.ndarray,
    onset_frame: int,
    impact_frame: int,
    mean: np.ndarray,
    std: np.ndarray,
    fs: int = 100,
    win_len_sec: float = 0.5,
    pre_onset_sec: float = 0.3,
    fall_class: int = 1,
) -> Tuple[bool, List[int]]:
    """Event-level eval for one fall trial using TFLite inference."""
    win_len = int(round(win_len_sec * fs))
    pre_onset_frames = int(round(pre_onset_sec * fs))

    windows = sliding_windows_until_impact(signal, impact_frame=impact_frame, win_len_frames=win_len, step=10)

    detect_frames: List[int] = []
    for w, end_frame in windows:
        x_float = preprocess_window_float(w, mean, std)  # (6,win_len,1) float32
        pred = predict_class_tflite(interpreter, in_detail, out_detail, x_float)
        if pred == fall_class:
            detect_frames.append(int(end_frame))

    lower = max(1, int(onset_frame) - int(pre_onset_frames))
    upper = int(impact_frame)
    detected = any(lower <= f <= upper for f in detect_frames)
    return detected, detect_frames


def evaluate_one_adl_trial_tflite(
    interpreter: tf.lite.Interpreter,
    in_detail: Dict,
    out_detail: Dict,
    signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    fs: int = 100,
    win_len_sec: float = 0.5,
    fall_class: int = 1,
) -> Tuple[bool, List[int]]:
    """Event-level eval for one ADL trial using TFLite inference."""
    win_len = int(round(win_len_sec * fs))
    windows = sliding_windows_full(signal, win_len_frames=win_len, step=10)

    detect_frames: List[int] = []
    for w, end_frame in windows:
        x_float = preprocess_window_float(w, mean, std)
        pred = predict_class_tflite(interpreter, in_detail, out_detail, x_float)
        if pred == fall_class:
            detect_frames.append(int(end_frame))

    detected = len(detect_frames) > 0
    return detected, detect_frames


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_dir", type=str, required=True, help="CSV directory containing SxxTyyRzz.csv")
    parser.add_argument("--label_path", type=str, required=True, help="Label excel path (e.g., SA37_label.xlsx)")
    parser.add_argument("--subject_prefix", type=str, required=True, help="Prefix used for csv naming (e.g., S37)")
    parser.add_argument("--tflite_path", type=str, required=True, help="Quantized TFLite model path (*.tflite)")
    parser.add_argument("--train_data_root", type=str, default="./data", help="Root containing train/X.npy for mean/std")
    parser.add_argument("--fs", type=int, default=100, help="Sampling rate Hz")
    parser.add_argument("--win_sec", type=float, default=0.5, help="Window length in seconds")
    parser.add_argument("--pre_onset_sec", type=float, default=0.4, help="Seconds before onset to start detection window")
    parser.add_argument("--fall_class", type=int, default=1, help="Fall class index (default 1)")
    parser.add_argument("--out_json", type=str, default="", help="Output json path (default: eval_results_<prefix>_tflite.json)")
    args = parser.parse_args()

    # Norm stats
    mean, std = compute_norm_stats_from_train(args.train_data_root)
    print("Norm stats loaded from train data:")
    print("mean:", mean)
    print("std :", std)

    # Labels
    labels = parse_labels(args.label_path, subject_prefix=args.subject_prefix)
    print(f"Found {len(labels)} fall trials in label file.")
    fall_csv_set = set(info["csv_name"] for info in labels)

    # TFLite interpreter
    interpreter = make_interpreter(args.tflite_path)
    in_detail, out_detail = get_io_details(interpreter)

    print("\n=== TFLite IO ===")
    print("Input :", in_detail["shape"], in_detail["dtype"], "quant:", in_detail.get("quantization", None))
    print("Output:", out_detail["shape"], out_detail["dtype"], "quant:", out_detail.get("quantization", None))

    # 1) Fall trials: TP/FN
    tp = fn = 0
    fall_trials_evaluated = 0
    per_trial_results: List[Dict] = []

    for info in labels:
        csv_path = os.path.join(args.subject_dir, info["csv_name"])
        print(
            f"\n=== Fall Trial: {info['csv_name']} "
            f"(task_id={info['task_id']}, trial_id={info['trial_id']}) ==="
        )
        if not os.path.exists(csv_path):
            print(f"  [WARN] CSV not found, skip: {csv_path}")
            continue

        signal = load_signal_from_csv(csv_path)

        detected, frames = evaluate_one_fall_trial_tflite(
            interpreter=interpreter,
            in_detail=in_detail,
            out_detail=out_detail,
            signal=signal,
            onset_frame=info["onset_frame"],
            impact_frame=info["impact_frame"],
            mean=mean,
            std=std,
            fs=args.fs,
            win_len_sec=args.win_sec,
            pre_onset_sec=args.pre_onset_sec,
            fall_class=args.fall_class,
        )

        fall_trials_evaluated += 1
        if detected:
            tp += 1
            print(f"  -> TP (detected in [onset-{args.pre_onset_sec}s, impact]). positive frames: {frames}")
        else:
            fn += 1
            print(f"  -> FN (no detection in [onset-{args.pre_onset_sec}s, impact]). positive frames: {frames}")

        per_trial_results.append(
            dict(
                csv=info["csv_name"],
                task_id=info["task_id"],
                trial_id=info["trial_id"],
                onset_frame=info["onset_frame"],
                impact_frame=info["impact_frame"],
                is_fall=True,
                detected=bool(detected),
                positive_frames=frames,
            )
        )

    # 2) ADL trials: TN/FP
    adl_tn = adl_fp = 0
    adl_trials_evaluated = 0

    if os.path.isdir(args.subject_dir):
        all_files = sorted(f for f in os.listdir(args.subject_dir) if f.lower().endswith(".csv"))
    else:
        all_files = []
        print(f"[WARN] subject_dir does not exist: {args.subject_dir}")

    for fname in all_files:
        if fname in fall_csv_set:
            continue

        parsed = parse_sensor_filename(fname)
        if parsed is None:
            print(f"[ADL] [Skip] Cannot parse filename: {fname}")
            continue

        csv_path = os.path.join(args.subject_dir, fname)
        print(f"\n=== ADL Trial: {fname} (task_id={parsed['task_id']}, trial_id={parsed['trial_id']}) ===")

        try:
            signal = load_signal_from_csv(csv_path)
        except Exception as e:
            print(f"  [ADL] Read failed, skip: {e}")
            continue

        detected, frames = evaluate_one_adl_trial_tflite(
            interpreter=interpreter,
            in_detail=in_detail,
            out_detail=out_detail,
            signal=signal,
            mean=mean,
            std=std,
            fs=args.fs,
            win_len_sec=args.win_sec,
            fall_class=args.fall_class,
        )

        adl_trials_evaluated += 1
        if detected:
            adl_fp += 1
            print(f"  -> FP (at least one fall prediction). positive frames: {frames}")
        else:
            adl_tn += 1
            print("  -> TN (no fall prediction).")

        per_trial_results.append(
            dict(
                csv=fname,
                task_id=parsed["task_id"],
                trial_id=parsed["trial_id"],
                onset_frame=None,
                impact_frame=None,
                is_fall=False,
                detected=bool(detected),
                positive_frames=frames,
            )
        )

    # 3) Summary metrics
    print("\n===== Summary (Per-trial metrics, TFLite) =====")
    print(f"Fall trials in label file      : {len(labels)}")
    print(f"Fall trials evaluated          : {fall_trials_evaluated}")
    print(f"TP (fall, detected)            : {tp}")
    print(f"FN (fall, not detected)        : {fn}")

    if (tp + fn) > 0:
        sensitivity = tp / (tp + fn)
        print(f"Sensitivity (TPR)              : {sensitivity:.4f}")
    else:
        sensitivity = None
        print("Sensitivity (TPR)              : N/A (no fall trial evaluated)")

    print("\nADL trials evaluated           :", adl_trials_evaluated)
    print(f"TN (ADL, no false alarm)       : {adl_tn}")
    print(f"FP (ADL, false alarm)          : {adl_fp}")

    if (adl_tn + adl_fp) > 0:
        specificity = adl_tn / (adl_tn + adl_fp)
        print(f"Specificity (TNR)              : {specificity:.4f}")
    else:
        specificity = None
        print("Specificity (TNR)              : N/A (no ADL trial evaluated)")

    # 4) Save per-trial details
    if args.out_json:
        results_path = args.out_json
    else:
        results_path = os.path.join(
            os.path.dirname(__file__), f"eval_results_{args.subject_prefix.lower()}_tflite.json"
        )

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(per_trial_results, f, ensure_ascii=False, indent=2)

    print(f"\nPer-trial results saved to: {results_path}")


if __name__ == "__main__":
    main()

#python evaluate_quantized.py --subject_dir "C:\ELEC\ELEC491\KFall\sensor_data\SA37" --label_path "C:\ELEC\ELEC491\KFall\label_data\SA37_label.xlsx" --subject_prefix S37 --tflite_path ./model_int8.tflite --train_data_root ./data
