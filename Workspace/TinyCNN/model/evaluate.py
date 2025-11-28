# evaluate.py
import os
import re
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import json

from model import TinyCNN
from utils import load_model


# =========================
# 标签解析相关
# =========================
def parse_task_id_from_cell(cell: str):
    """从 'F01 (20)' 这种单元格中提取括号里的 task_id（int）"""
    if pd.isna(cell):
        return None
    m = re.search(r"\((\d+)\)", str(cell))
    return int(m.group(1)) if m else None


def load_label_table(xlsx_path: str) -> pd.DataFrame:
    """
    解析 SA06_label.xlsx => [task_id, trial_id, onset_frame, impact_frame]
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    col_task = next((c for c in df.columns if "Task Code" in c or "Task" in c), None)
    col_trial = next((c for c in df.columns if "Trial" in c and "ID" in c), None)
    col_onset = next((c for c in df.columns if "onset" in c.lower()), None)
    col_impact = next((c for c in df.columns if "impact" in c.lower()), None)
    if not all([col_task, col_trial, col_onset, col_impact]):
        raise ValueError(f"无法在 {xlsx_path} 中识别关键列，表头：{df.columns.tolist()}")

    # 用上一行 Task Code 填充空行
    df[col_task] = df[col_task].ffill()
    # 从 'F01 (20)' 提取 20 作为 task_id
    df["task_id"] = df[col_task].apply(parse_task_id_from_cell)

    out = df[["task_id", col_trial, col_onset, col_impact]].copy()
    out.columns = ["task_id", "trial_id", "onset_frame", "impact_frame"]

    out = out.dropna(subset=["task_id", "trial_id", "onset_frame", "impact_frame"])
    out["task_id"] = out["task_id"].astype(int)
    out["trial_id"] = out["trial_id"].astype(int)
    out["onset_frame"] = out["onset_frame"].astype(int)
    out["impact_frame"] = out["impact_frame"].astype(int)
    return out


def parse_labels(
    excel_path: str,
    subject_prefix: str = "S06",
):
    """
    使用 label 表生成：
        task_id, trial_id, onset_frame, impact_frame, csv_name
    其中 csv_name 形如 S06T20R01.csv
    """
    df = load_label_table(excel_path)

    entries = []
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
# 归一化参数（与训练保持一致）
# =========================
def compute_norm_stats_from_train(data_root: str):
    """
    仿照 dataset.py 里 get_loaders 的做法：
    从 data_root/train/X.npy 计算全局 mean/std（按 N,T 聚合，按通道算）。
    """
    train_x_path = os.path.join(data_root, "train", "X.npy")
    if not os.path.exists(train_x_path):
        raise FileNotFoundError(
            f"未找到训练集 X.npy：{train_x_path}，请确认 --train_data_root 设置正确"
        )

    train_X = np.load(train_x_path)  # 形状应为 (N, T, C)
    if train_X.ndim != 3:
        raise ValueError(f"train X.npy 形状异常：{train_X.shape}，期望 (N, T, C)")

    mean = train_X.mean(axis=(0, 1)).astype(np.float32)  # (C,)
    std = train_X.std(axis=(0, 1)).astype(np.float32)    # (C,)
    std = np.where(std == 0, 1.0, std)
    return mean, std


# =========================
# 传感器 / 文件解析
# =========================
def load_signal_from_csv(csv_path: str) -> np.ndarray:
    """
    从一个 S06TxxRyy.csv 读取 6 轴信号：
        AccX, AccY, AccZ, GyrX, GyrY, GyrZ

    返回形状 (N_frames, 6) 的 numpy 数组。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # 尽量兼容列名
    def pick(col_name: str) -> str:
        candidates = [c for c in df.columns if c.lower().startswith(col_name.lower())]
        if not candidates:
            raise KeyError(f"在 {csv_path} 中找不到列 '{col_name}'（不区分大小写）")
        return candidates[0]

    accx = pick("AccX")
    accy = pick("AccY")
    accz = pick("AccZ")
    gyrx = pick("GyrX")
    gyry = pick("GyrY")
    gyrz = pick("GyrZ")

    sig = df[[accx, accy, accz, gyrx, gyry, gyrz]].values.astype(np.float32)
    return sig


def parse_sensor_filename(fname: str):
    """
    解析形如 S06T20R01.csv => subj=6, task_id=20, trial_id=1
    """
    m = re.match(r"S(?P<subj>\d+)T(?P<task>\d{2})R(?P<rep>\d{2})\.csv$", fname, re.IGNORECASE)
    if not m:
        return None
    return {
        "subj": int(m.group("subj")),
        "task_id": int(m.group("task")),
        "trial_id": int(m.group("rep")),
    }


# =========================
# 滑窗生成
# =========================
def sliding_windows_until_impact(
    signal: np.ndarray,
    impact_frame: int,
    win_len_frames: int,
    step: int = 10,
) -> List[Tuple[np.ndarray, int]]:
    """
    基于帧滑窗直到 impact_frame。

    参数：
        signal: (N, 6)
        impact_frame: 1-based 帧号（与 label 对齐）
        win_len_frames: 窗口长度（帧数），例如 50
        step: 步长（帧），默认 10 帧 ≈ 0.1s (100Hz)

    返回列表 [ (window, end_frame), ... ]，
    其中 window 形状为 (win_len_frames, 6)，end_frame 为 1-based 结束帧号。
    """
    N = signal.shape[0]
    windows = []

    impact = min(impact_frame, N)

    for end_frame in range(win_len_frames, impact + 1, step):
        start_idx = end_frame - win_len_frames
        end_idx = end_frame

        window = signal[start_idx:end_idx, :]
        if window.shape[0] != win_len_frames:
            continue
        windows.append((window, end_frame))

    return windows


def sliding_windows_full(
    signal: np.ndarray,
    win_len_frames: int,
    step: int = 10,
) -> List[Tuple[np.ndarray, int]]:
    """
    针对 ADL：在整个 trial 上做全程滑窗。
    返回 [ (window, end_frame), ... ]，end_frame 为 1-based。
    """
    N = signal.shape[0]
    windows = []
    for end_frame in range(win_len_frames, N + 1, step):
        start_idx = end_frame - win_len_frames
        end_idx = end_frame
        window = signal[start_idx:end_idx, :]
        if window.shape[0] != win_len_frames:
            continue
        windows.append((window, end_frame))
    return windows


# =========================
# 单 trial 评估（带归一化）
# =========================
def evaluate_one_fall_trial(
    model: torch.nn.Module,
    device: torch.device,
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
    """
    对单个跌倒 trial 做评估。

    TP 判定规则：
        只要存在至少一个窗口：
            - 被模型判为 fall_class
            - 窗口结束帧 end_frame 满足：
                onset_frame - pre_onset_sec * fs <= end_frame <= impact_frame
        则该 trial 视为 detected=True (TP)。
    """
    model.eval()

    win_len = int(round(win_len_sec * fs))            # 0.5s -> 50
    pre_onset_frames = int(round(pre_onset_sec * fs)) # 0.3s -> 30

    # 归一化，与训练保持一致
    std_safe = np.where(std == 0, 1.0, std)
    signal = (signal - mean) / std_safe

    windows = sliding_windows_until_impact(
        signal, impact_frame=impact_frame, win_len_frames=win_len, step=10
    )

    detect_frames: List[int] = []
    with torch.no_grad():
        for w, end_frame in windows:
            # TinyCNN 支持 (B, 6, 50) 作为输入
            x = torch.from_numpy(w.T).unsqueeze(0).float().to(device)  # (1, 6, 50)
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            if pred == fall_class:
                detect_frames.append(end_frame)

    lower = max(1, onset_frame - pre_onset_frames)
    upper = impact_frame
    detected = any(lower <= f <= upper for f in detect_frames)

    return detected, detect_frames


def evaluate_one_adl_trial(
    model: torch.nn.Module,
    device: torch.device,
    signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    fs: int = 100,
    win_len_sec: float = 0.5,
    fall_class: int = 1,
) -> Tuple[bool, List[int]]:
    """
    对单个 ADL trial（非跌倒）做评估。

    规则：
        - 只要全程任何一个窗口被判为 fall_class，就认为该 trial 有一次“误报”（FP）
          => detected=True
        - 如果全程没有 fall 判定 => detected=False（TN）
    """
    model.eval()

    win_len = int(round(win_len_sec * fs))

    std_safe = np.where(std == 0, 1.0, std)
    signal = (signal - mean) / std_safe

    windows = sliding_windows_full(signal, win_len_frames=win_len, step=10)

    detect_frames: List[int] = []
    with torch.no_grad():
        for w, end_frame in windows:
            x = torch.from_numpy(w.T).unsqueeze(0).float().to(device)
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            if pred == fall_class:
                detect_frames.append(end_frame)

    detected = len(detect_frames) > 0
    return detected, detect_frames


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject_dir",
        type=str,
        default=r"C:\\ELEC\\ELEC491\\KFall\\sensor_data\\SA37",
        help="SA06 对应的 csv 目录（包含 S06TxxRyy.csv）",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=r"C:\\ELEC\\ELEC491\\KFall\\label_data\\SA37_label.xlsx",
        help="SA06 的标注文件路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model_epoch200.pth",
        help="TinyCNN 权重路径",
    )
    parser.add_argument(
        "--train_data_root",
        type=str,
        default="./data",   # 里面要有 train/X.npy
        help="训练数据根目录（包含 train/X.npy，用于计算 mean/std）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto/cpu/cuda/mps",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=100,
        help="采样频率 Hz（KFall 是 100Hz）",
    )
    parser.add_argument(
        "--win_sec",
        type=float,
        default=0.5,
        help="窗口长度（秒），默认 0.5",
    )
    parser.add_argument(
        "--pre_onset_sec",
        type=float,
        default=0.4,
        help="Fall_onset_frame 之前多少秒开始算检测窗口，默认 0.3",
    )
    parser.add_argument(
        "--fall_class",
        type=int,
        default=1,
        help="fall 类别标签（TinyCNN 输出中的 index）",
    )
    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("Using device:", device)

    # 归一化参数：与训练保持一致
    mean, std = compute_norm_stats_from_train(args.train_data_root)
    print("Norm stats (mean/std) loaded from train data:")
    print("mean:", mean)
    print("std :", std)

    # 解析 label（跌倒 trial）
    labels = parse_labels(args.label_path, subject_prefix="S37")
    print(f"Found {len(labels)} fall trials in label file.")

    # 方便后面判断哪些文件是 fall trial
    fall_csv_set = set(info["csv_name"] for info in labels)

    # 加载模型
    model = load_model(TinyCNN, path=args.model_path)
    model.to(device)

    # -----------------------------
    # 1) 跌倒 trial：统计 TP / FN
    # -----------------------------
    tp = fn = 0
    fall_trials_evaluated = 0
    per_trial_results = []

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
        detected, frames = evaluate_one_fall_trial(
            model=model,
            device=device,
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
            print(
                f"  -> TP (detected between onset-"
                f"{args.pre_onset_sec}s and impact). "
                f"positive frames: {frames}"
            )
        else:
            fn += 1
            print(
                f"  -> FN (no detection between onset-"
                f"{args.pre_onset_sec}s and impact). "
                f"positive frames: {frames}"
            )

        per_trial_results.append(
            dict(
                csv=info["csv_name"],
                task_id=info["task_id"],
                trial_id=info["trial_id"],
                onset_frame=info["onset_frame"],
                impact_frame=info["impact_frame"],
                is_fall=True,
                detected=detected,
                positive_frames=frames,
            )
        )

    # -----------------------------
    # 2) ADL trial：统计 TN / FP
    # -----------------------------
    adl_tn = adl_fp = 0
    adl_trials_evaluated = 0

    if os.path.isdir(args.subject_dir):
        all_files = sorted(
            f for f in os.listdir(args.subject_dir) if f.lower().endswith(".csv")
        )
    else:
        all_files = []
        print(f"[WARN] subject_dir 不存在：{args.subject_dir}")

    for fname in all_files:
        if fname in fall_csv_set:
            continue  # 已经作为跌倒 trial 评估过

        parsed = parse_sensor_filename(fname)
        if parsed is None:
            print(f"[ADL] [跳过] 无法解析文件名：{fname}")
            continue

        csv_path = os.path.join(args.subject_dir, fname)
        print(f"\n=== ADL Trial: {fname} (task_id={parsed['task_id']}, trial_id={parsed['trial_id']}) ===")

        try:
            signal = load_signal_from_csv(csv_path)
        except Exception as e:
            print(f"  [ADL] 读取失败，跳过：{e}")
            continue

        detected, frames = evaluate_one_adl_trial(
            model=model,
            device=device,
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
                detected=detected,
                positive_frames=frames,
            )
        )

    # -----------------------------
    # 3) 汇总指标：Sensitivity / Specificity
    # -----------------------------
    print("\n===== Summary (Per-trial metrics) =====")
    print(f"Fall trials in label file      : {len(labels)}")
    print(f"Fall trials evaluated          : {fall_trials_evaluated}")
    print(f"TP (fall, detected)            : {tp}")
    print(f"FN (fall, not detected)        : {fn}")

    if tp + fn > 0:
        sensitivity = tp / (tp + fn)
        print(f"Sensitivity (TPR)              : {sensitivity:.4f}")
    else:
        print("Sensitivity (TPR)              : N/A (no fall trial evaluated)")

    print("\nADL trials evaluated           :", adl_trials_evaluated)
    print(f"TN (ADL, no false alarm)       : {adl_tn}")
    print(f"FP (ADL, false alarm)          : {adl_fp}")

    if adl_tn + adl_fp > 0:
        specificity = adl_tn / (adl_tn + adl_fp)
        print(f"Specificity (TNR)              : {specificity:.4f}")
    else:
        print("Specificity (TNR)              : N/A (no ADL trial evaluated)")

    # -----------------------------
    # 4) 保存每个 trial 的详细结果，方便后续可视化
    # -----------------------------
    results_path = os.path.join(os.path.dirname(__file__), "eval_results_sa37.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(per_trial_results, f, ensure_ascii=False, indent=2)
    print(f"\nPer-trial results saved to: {results_path}")


if __name__ == "__main__":
    main()
