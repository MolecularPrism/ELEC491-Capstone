# export_tflite_int8.py
# Export a full-integer INT8 TFLite model from a trained Keras TinyCNN.
# - Loads Keras weights (*.weights.h5)
# - Builds representative dataset from data_root/train/X.npy
# - Uses training-set mean/std to match normalization
# - Produces an INT8 TFLite model (input/output int8 by default)

from __future__ import annotations

import os
import json
import argparse
from typing import Iterator, List, Tuple

import numpy as np
import tensorflow as tf

from model import TinyCNN


def compute_norm_stats_from_train(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std from data_root/train/X.npy with shape (N,T,C)."""
    train_x_path = os.path.join(data_root, "train", "X.npy")
    if not os.path.exists(train_x_path):
        raise FileNotFoundError(f"train X.npy not found: {train_x_path}")

    train_X = np.load(train_x_path)  # (N,T,C)
    if train_X.ndim != 3:
        raise ValueError(f"Expected train_X shape (N,T,C), got {train_X.shape}")

    mean = train_X.mean(axis=(0, 1)).astype(np.float32)  # (C,)
    std = train_X.std(axis=(0, 1)).astype(np.float32)    # (C,)
    std = np.where(std == 0, 1.0, std).astype(np.float32)
    return mean, std


def preprocess_window(x_tc: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Match training preprocessing:
      x_tc: (T,C) -> normalize -> transpose -> (C,T) -> expand -> (C,T,1)
    Returns float32 NHWC sample: (6,50,1)
    """
    x = x_tc.astype(np.float32)
    x = (x - mean) / std
    x = x.T  # (C,T)
    x = np.expand_dims(x, axis=-1)  # (C,T,1)
    return x.astype(np.float32)


def build_representative_dataset(
    data_root: str,
    num_samples: int,
    seed: int,
) -> Iterator[List[np.ndarray]]:
    """
    Representative dataset generator for TFLite calibration.
    Yields a list containing one input array shaped (1,6,50,1), float32.
    """
    train_x_path = os.path.join(data_root, "train", "X.npy")
    train_X = np.load(train_x_path)  # (N,T,C)

    mean, std = compute_norm_stats_from_train(data_root)

    rng = np.random.default_rng(seed)
    n = train_X.shape[0]
    k = min(int(num_samples), int(n))
    idxs = rng.choice(n, size=k, replace=False)

    for i in idxs:
        x_tc = train_X[i]  # (T,C)
        x = preprocess_window(x_tc, mean, std)  # (6,50,1)
        x = np.expand_dims(x, axis=0)  # (1,6,50,1)
        yield [x]


def load_keras_model(weights_path: str) -> tf.keras.Model:
    """Build TinyCNN, create variables, then load weights."""
    model = TinyCNN(num_classes=2)
    dummy = tf.zeros((1, 6, 50, 1), dtype=tf.float32)
    _ = model(dummy, training=False)
    model.load_weights(weights_path)
    return model


def export_int8_tflite(
    model: tf.keras.Model,
    rep_data_gen,
    out_path: str,
    input_type: str = "int8",
    output_type: str = "int8",
) -> bytes:
    """
    Export full-integer INT8 TFLite.
    input_type/output_type: 'int8' or 'uint8' or 'float32'
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen

    # Full integer quantization for builtin INT8 kernels
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    type_map = {
        "int8": tf.int8,
        "uint8": tf.uint8,
        "float32": tf.float32,
    }
    if input_type not in type_map or output_type not in type_map:
        raise ValueError("input_type/output_type must be one of: int8, uint8, float32")

    converter.inference_input_type = type_map[input_type]
    converter.inference_output_type = type_map[output_type]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def inspect_tflite(tflite_path: str) -> dict:
    """Load the TFLite model and return basic IO quantization info."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    info = {
        "inputs": [],
        "outputs": [],
    }

    for d in in_details:
        info["inputs"].append(
            {
                "name": d.get("name"),
                "shape": list(d.get("shape", [])),
                "dtype": str(d.get("dtype")),
                "quantization": list(d.get("quantization", (0.0, 0))),
                "quantization_parameters": {
                    "scales": np.array(d.get("quantization_parameters", {}).get("scales", [])).tolist(),
                    "zero_points": np.array(d.get("quantization_parameters", {}).get("zero_points", [])).tolist(),
                    "quantized_dimension": int(d.get("quantization_parameters", {}).get("quantized_dimension", 0)),
                },
            }
        )
    for d in out_details:
        info["outputs"].append(
            {
                "name": d.get("name"),
                "shape": list(d.get("shape", [])),
                "dtype": str(d.get("dtype")),
                "quantization": list(d.get("quantization", (0.0, 0))),
                "quantization_parameters": {
                    "scales": np.array(d.get("quantization_parameters", {}).get("scales", [])).tolist(),
                    "zero_points": np.array(d.get("quantization_parameters", {}).get("zero_points", [])).tolist(),
                    "quantized_dimension": int(d.get("quantization_parameters", {}).get("quantized_dimension", 0)),
                },
            }
        )

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to Keras weights (*.weights.h5)")
    parser.add_argument("--train_data_root", type=str, default="./data", help="Root containing train/X.npy")
    parser.add_argument("--out", type=str, default="model_int8.tflite", help="Output TFLite path")
    parser.add_argument("--num_calib_samples", type=int, default=1000, help="Num samples for calibration")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for sampling calibration windows")
    parser.add_argument("--input_type", type=str, default="int8", choices=["int8", "uint8", "float32"])
    parser.add_argument("--output_type", type=str, default="int8", choices=["int8", "uint8", "float32"])
    parser.add_argument("--save_io_json", action="store_true", help="Save IO quantization info to JSON")
    args = parser.parse_args()

    print("Loading model weights:", args.weights)
    model = load_keras_model(args.weights)

    print("Building representative dataset from:", args.train_data_root)
    rep_gen = lambda: build_representative_dataset(
        data_root=args.train_data_root,
        num_samples=args.num_calib_samples,
        seed=args.seed,
    )

    print(f"Exporting INT8 TFLite to: {args.out}")
    _ = export_int8_tflite(
        model=model,
        rep_data_gen=rep_gen,
        out_path=args.out,
        input_type=args.input_type,
        output_type=args.output_type,
    )

    info = inspect_tflite(args.out)
    print("\n=== TFLite IO Details ===")
    print(json.dumps(info, indent=2))

    if args.save_io_json:
        json_path = os.path.splitext(args.out)[0] + "_io.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print("Saved IO info to:", json_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
    

#python export_tflite_int8.py --weights ./model_best.weights.h5 --train_data_root ./data --out ./model_int8.tflite --num_calib_samples 1000 --input_type int8 --output_type int8