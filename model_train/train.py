# train.py
# TensorFlow/Keras training script ported from the PyTorch version.
# It trains TinyCNN on data_dir/{train,val}/X.npy and y.npy.
# Notes:
# - Uses NHWC input layout: (B, 6, 50, 1)
# - X.npy is expected to be (N, T, C) where typically T=50, C=6
# - Normalization stats are computed from the training set: mean/std over axes (0,1) -> per-channel

from __future__ import annotations

import os
import argparse
import numpy as np
import tensorflow as tf

from model import TinyCNN


def pick_device(pref: str) -> str:
    """Pick a TF device string. This is best-effort; TF will still place ops automatically."""
    pref = str(pref).lower()
    if pref == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        return "/GPU:0" if len(gpus) > 0 else "/CPU:0"
    if pref in ("cpu", "/cpu:0"):
        return "/CPU:0"
    if pref in ("gpu", "cuda", "/gpu:0"):
        return "/GPU:0"
    # Allow advanced users to pass a raw TF device string.
    return pref


def compute_norm_stats(train_x: np.ndarray) -> dict:
    """
    Compute per-channel normalization stats to match the PyTorch code:
    mean = train_X.mean(axis=(0,1)), std = train_X.std(axis=(0,1)).
    train_x: (N, T, C)
    """
    mean = train_x.mean(axis=(0, 1)).astype(np.float32)  # (C,)
    std = train_x.std(axis=(0, 1)).astype(np.float32)    # (C,)
    std = np.where(std == 0, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def make_tf_datasets(
    data_dir: str,
    batch_size: int,
    normalize: bool = True,
    drop_remainder: bool = True,
):
    """
    Build tf.data datasets equivalent to the PyTorch DataLoader behavior.
    Returns: train_ds, val_ds, norm_stats
    """
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")

    train_x = np.load(os.path.join(train_root, "X.npy"))  # (N, T, C)
    train_y = np.load(os.path.join(train_root, "y.npy"))  # (N,)
    val_x = np.load(os.path.join(val_root, "X.npy"))      # (N, T, C)
    val_y = np.load(os.path.join(val_root, "y.npy"))      # (N,)

    norm_stats = compute_norm_stats(train_x) if normalize else None

    mean = norm_stats["mean"] if norm_stats is not None else None
    std = norm_stats["std"] if norm_stats is not None else None

    def preprocess(x, y):
        """
        x: (T, C) float32
        Convert to NHWC for Keras: (H=6, W=50, C=1).
        We match the PyTorch dataset behavior:
          - normalize per channel (broadcast over T)
          - transpose (T,C)->(C,T)
          - add channel dim -> (C,T,1)
        """
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int32)

        if mean is not None and std is not None:
            m = tf.constant(mean, dtype=tf.float32)  # (C,)
            s = tf.constant(std, dtype=tf.float32)   # (C,)
            x = (x - m) / s

        # (T, C) -> (C, T)
        x = tf.transpose(x, perm=[1, 0])

        # (C, T) -> (C, T, 1) i.e., (6, 50, 1)
        x = tf.expand_dims(x, axis=-1)
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(buffer_size=len(train_x), reshuffle_each_iteration=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=drop_remainder)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, norm_stats


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss ported from the PyTorch implementation.

    PyTorch version:
      ce_loss = cross_entropy(logits, targets, reduction="none")
      pt = exp(-ce_loss)
      focal = ((1-pt)**gamma) * ce_loss
      alpha_t = (1-alpha) for negatives, alpha for positives (targets==1)
      return mean/sum
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean", name="focal_loss"):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction_mode = str(reduction).lower()

    def call(self, y_true, y_pred):
        """
        y_true: (B,) int32 labels
        y_pred: (B, C) logits
        """
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        logits = tf.cast(y_pred, tf.float32)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)  # (B,)
        pt = tf.exp(-ce)  # (B,)
        focal = tf.pow(1.0 - pt, self.gamma) * ce

        # Match the PyTorch alpha scheme: pos_class assumed to be label 1 here.
        # Negative weight: (1-alpha), Positive weight (label==1): alpha
        alpha_t = tf.ones_like(focal, dtype=tf.float32) * (1.0 - self.alpha)
        alpha_t = tf.where(tf.equal(y_true, 1), tf.ones_like(alpha_t) * self.alpha, alpha_t)
        focal = alpha_t * focal

        if self.reduction_mode == "sum":
            return tf.reduce_sum(focal)
        # Default "mean"
        return tf.reduce_mean(focal)


@tf.function
def train_step(model, optimizer, loss_fn, xb, yb):
    """One training step with GradientTape."""
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)
        loss = loss_fn(yb, logits)

        # If loss_fn returns scalar, keep it. If it returns vector, reduce mean.
        if tf.rank(loss) > 0:
            loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def evaluate(model, dataset, loss_fn, pos_class: int = 1):
    """
    Return: avg_loss, sensitivity, specificity, (tp, fp, tn, fn)
    Matches the PyTorch evaluate() logic with argmax predictions.
    """
    total_loss = 0.0
    n_batches = 0

    tp = fp = tn = fn = 0

    for xb, yb in dataset:
        logits = model(xb, training=False)
        loss = loss_fn(yb, logits)
        loss_val = float(loss.numpy()) if np.isscalar(loss.numpy()) else float(np.mean(loss.numpy()))
        total_loss += loss_val
        n_batches += 1

        preds = tf.argmax(logits, axis=1, output_type=tf.int32)

        yb_i = tf.cast(yb, tf.int32)
        pos = tf.equal(yb_i, pos_class)
        neg = tf.logical_not(pos)
        pred_pos = tf.equal(preds, pos_class)
        pred_neg = tf.logical_not(pred_pos)

        tp += int(tf.reduce_sum(tf.cast(tf.logical_and(pred_pos, pos), tf.int32)).numpy())
        fp += int(tf.reduce_sum(tf.cast(tf.logical_and(pred_pos, neg), tf.int32)).numpy())
        tn += int(tf.reduce_sum(tf.cast(tf.logical_and(pred_neg, neg), tf.int32)).numpy())
        fn += int(tf.reduce_sum(tf.cast(tf.logical_and(pred_neg, pos), tf.int32)).numpy())

    eps = 1e-12
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, sensitivity, specificity, (tp, fp, tn, fn)


def save_weights_safely(model: tf.keras.Model, path: str):
    """
    Save weights to a Keras-friendly file.
    If the user provides a .pth path (from PyTorch habit), we still save TF weights
    by converting to a .weights.h5 file name.
    """
    base, ext = os.path.splitext(path)
    if ext.lower() in (".h5", ".keras", ".ckpt"):
        weights_path = path if ext.lower() != ".keras" else base + ".weights.h5"
    else:
        weights_path = base + ".weights.h5"
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    model.save_weights(weights_path)
    return weights_path


def train(
    model,
    train_ds,
    val_ds,
    loss_fn,
    optimizer,
    num_epochs: int = 5,
    device_str: str = "/CPU:0",
    save_interval: int = 0,
    save_path: str = "model.weights.h5",
    save_best: bool = True,
    eval_interval: int = 1,
    pos_class: int = 1,
):
    best_metric = -1.0

    # Build model variables by running one forward pass (ensures save_weights works immediately).
    for xb, _ in train_ds.take(1):
        _ = model(xb, training=False)

    with tf.device(device_str):
        for epoch in range(num_epochs):
            running_loss = 0.0
            n_batches = 0

            for xb, yb in train_ds:
                loss = train_step(model, optimizer, loss_fn, xb, yb)
                running_loss += float(loss.numpy())
                n_batches += 1

            train_loss = running_loss / max(1, n_batches)
            msg = f"Epoch [{epoch+1}/{num_epochs}]  train_loss={train_loss:.4f}"

            # Validation
            if val_ds is not None and ((epoch + 1) % eval_interval == 0):
                val_loss, sens, spec, (tp, fp, tn, fn) = evaluate(
                    model, val_ds, loss_fn, pos_class=pos_class
                )
                msg += (
                    f"  |  val_loss={val_loss:.4f}  "
                    f"sensitivity={sens:.4f}  specificity={spec:.4f}  "
                    f"[TP={tp} FP={fp} TN={tn} FN={fn}]"
                )

                # The original PyTorch code has "save_best" logic commented out.
                # We keep it available here (disabled by default if you prefer).
                if save_best and sens > best_metric:
                    best_metric = sens
                    best_path = os.path.splitext(save_path)[0] + "_best.weights.h5"
                    save_weights_safely(model, best_path)

            print(msg)

            # Save checkpoint at intervals
            if save_interval > 0 and ((epoch + 1) % save_interval == 0):
                ckpt_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.weights.h5"
                save_weights_safely(model, ckpt_path)

        # Final eval
        if val_ds is not None:
            val_loss, sens, spec, (tp, fp, tn, fn) = evaluate(
                model, val_ds, loss_fn, pos_class=pos_class
            )
            print(
                f"[Final] val_loss={val_loss:.4f}  sensitivity={sens:.4f}  specificity={spec:.4f}  "
                f"[TP={tp} FP={fp} TN={tn} FN={fn}]"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/gpu or TF device string")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)  # Kept for CLI compatibility (not used by Adam/AdamW)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--data_dir", type=str, default="./data", help="dir containing train/ and val/")
    # Keep a similar flag name, but also allow turning it off.
    parser.add_argument("--normalize", action="store_true", default=True, help="use train stats for z-score normalization")
    parser.add_argument("--no_normalize", action="store_false", dest="normalize", help="disable normalization")
    parser.add_argument("--save_path", type=str, default="model.pth")  # Accept PyTorch-like name; will save .weights.h5
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--pos_class", type=int, default=1, help="positive class label (default 1=fall)")
    # Optional: expose focal loss params for easy tuning
    parser.add_argument("--focal_alpha", type=float, default=0.18)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    args = parser.parse_args()

    device_str = pick_device(args.device)
    print("Using device:", device_str)

    # Data
    train_ds, val_ds, _ = make_tf_datasets(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        normalize=args.normalize,
        drop_remainder=True,  # Match PyTorch: drop_last=True
    )

    # Model
    model = TinyCNN(num_classes=2)

    # Loss & optimizer
    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="mean")

    # Match PyTorch "Adam(..., weight_decay=...)" behavior.
    # Prefer AdamW when weight_decay > 0, else Adam.
    if args.weight_decay > 0:
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=args.lr, weight_decay=args.weight_decay
            )
        except Exception:
            # Fallback: Adam without decoupled weight decay.
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Train
    train(
        model,
        train_ds,
        val_ds,
        loss_fn,
        optimizer,
        num_epochs=args.epochs,
        device_str=device_str,
        save_interval=args.save_interval,
        save_path=args.save_path,
        save_best=args.save_best,
        eval_interval=args.eval_interval,
        pos_class=args.pos_class,
    )

    # Save final model weights
    final_path = save_weights_safely(model, args.save_path)
    print("Saved weights to:", final_path)


if __name__ == "__main__":
    main()


# python train.py --data_dir ./data --epochs 200 --batch_size 512 --lr 0.0005
