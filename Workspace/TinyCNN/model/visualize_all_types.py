# visualize_all_types.py
import os
import json
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================
# 读取 CSV（原始六轴）
# =============================
def load_signal_from_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    needed = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}，实际列：{df.columns.tolist()}")

    return df


# =============================
# 画单个 trial
# =============================
def plot_trial_full(
    df: pd.DataFrame,
    info: Dict,
    win_len_frames: int,
    out_path: str,
    fs: int = 100,
):
    cols = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
    n_axis = len(cols)
    n = len(df)
    x = np.arange(n)

    is_fall = info["is_fall"]
    detected = info["detected"]
    onset_frame = info.get("onset_frame", None)
    impact_frame = info.get("impact_frame", None)
    pos_frames: List[int] = info.get("positive_frames", [])

    fig, axes = plt.subplots(n_axis, 1, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(x, df[col].to_numpy(), label=col)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

        # onset / impact（跌倒 trial）
        if is_fall:
            if onset_frame:
                ax.axvline(onset_frame - 1, color="orange", linestyle="--", linewidth=1.2)
            if impact_frame:
                ax.axvline(impact_frame - 1, color="red", linestyle="--", linewidth=1.2)

        # 高亮 positive window
        for end_frame in pos_frames:
            start_idx = max(0, end_frame - win_len_frames)
            end_idx = min(n, end_frame)
            ax.axvspan(start_idx, end_idx, color="red", alpha=0.15)

    title = f"{info['csv']} | "
    if is_fall:
        title += f"FALL | {'TP' if detected else 'FN'}"
    else:
        title += f"ADL | {'FP' if detected else 'TN'}"

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================
# 主流程
# =============================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject_dir",
        type=str,
        required=True,
        help="某个 SAxx 的 sensor_data 目录路径",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        required=True,
        help="evaluate.py 输出的 JSON 文件",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="viz_all",
        help="图片输出目录",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--win_sec",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=["TP", "TN", "FP", "FN"],
        help="要可视化哪些类型，可选: TP TN FP FN",
    )

    args = parser.parse_args()

    win_len_frames = int(round(args.win_sec * args.fs))

    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    type_set = set(t.upper() for t in args.types)

    print(f"可视化类型: {type_set}")

    count = {t: 0 for t in ["TP", "TN", "FP", "FN"]}

    for info in results:
        csv_name = info["csv"]
        is_fall = info["is_fall"]
        detected = info["detected"]

        if is_fall:
            kind = "TP" if detected else "FN"
        else:
            kind = "FP" if detected else "TN"

        if kind not in type_set:
            continue

        csv_path = os.path.join(args.subject_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing CSV: {csv_path}")
            continue

        try:
            df = load_signal_from_csv(csv_path)
        except Exception as e:
            print("[WARN]", csv_name, "读取失败", e)
            continue

        out_path = os.path.join(args.output_dir, f"{kind}_{csv_name.replace('.csv','.png')}")
        print(f"Plotting {kind}: {csv_name}")
        plot_trial_full(df, info, win_len_frames, out_path, fs=args.fs)

        count[kind] += 1

    print("\n=== DONE ===")
    for k, v in count.items():
        print(f"{k}: {v} plotted")
    print(f"Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# python visualize_all_types.py --subject_dir "C:\ELEC\ELEC491\KFall\sensor_data\SA37" --results_json eval_results_sa37.json --output_dir figs_sa37 --types TP FN FP TN   