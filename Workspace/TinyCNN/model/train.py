# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import TinyCNN  
from dataset import get_loaders
from utils import save_model

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B, C] output of model
        targets: [B] class labels
        """
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none", weight=None
        )  
        pt = torch.exp(-ce_loss)  
        #pt = softmax(logits)[range(B), targets]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.ones_like(targets, dtype=torch.float, device=targets.device) * (1.0 - self.alpha)
            alpha_t[targets == 1] = self.alpha
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss



def evaluate(model, loader, criterion, device="cpu", pos_class: int = 1):
    """
    return window-level loss, sensitivity, specificity, (tp, fp, tn, fn)
    """
    model.eval()
    total_loss = 0.0

    tp = fp = tn = fn = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)

            pos = (yb == pos_class)
            neg = ~pos
            pred_pos = (preds == pos_class)
            pred_neg = ~pred_pos

            tp += (pred_pos & pos).sum().item()
            fp += (pred_pos & neg).sum().item()
            tn += (pred_neg & neg).sum().item()
            fn += (pred_neg & pos).sum().item()

    eps = 1e-12
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    avg_loss = total_loss / max(1, len(loader))

    return avg_loss, sensitivity, specificity, (tp, fp, tn, fn)


def train(model,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          num_epochs=5,
          device="cpu",
          save_interval=0,
          save_path="model.pth",
          save_best=True,
          eval_interval=1,
          pos_class: int = 1):
    model.to(device)

    best_metric = -1.0   # for saving best model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        msg = f"Epoch [{epoch+1}/{num_epochs}]  train_loss={train_loss:.4f}"

        # val
        if val_loader is not None and ((epoch + 1) % eval_interval == 0):
            val_loss, sens, spec, (tp, fp, tn, fn) = evaluate(
                model, val_loader, criterion, device, pos_class=pos_class
            )
            msg += (f"  |  val_loss={val_loss:.4f}  "
                    f"sensitivity={sens:.4f}  specificity={spec:.4f}  "
                    f"[TP={tp} FP={fp} TN={tn} FN={fn}]")

            # if save_best and sens > best_metric:
                #best_metric = sens
                #best_path = os.path.splitext(save_path)[0] + "_best.pth"
                #save_model(model, best_path)

        print(msg)

        # Save checkpoint at intervals
        if save_interval > 0 and ((epoch + 1) % save_interval == 0):
            ckpt_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.pth"
            save_model(model, ckpt_path)

    # Evaluate once more after training ends
    if val_loader is not None:
        val_loss, sens, spec, (tp, fp, tn, fn) = evaluate(
            model, val_loader, criterion, device, pos_class=pos_class
        )
        print(f"[Final] val_loss={val_loss:.4f}  sensitivity={sens:.4f}  specificity={spec:.4f}  "
              f"[TP={tp} FP={fp} TN={tn} FN={fn}]")


def pick_device(pref: str):
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--data_dir", type=str, default="./data", help="dir containing train/ and val/")
    parser.add_argument("--normalize", action="store_true", default=True, help="use training set stats for z-score normalization")
    parser.add_argument("--save_path", type=str, default="model.pth")
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--pos_class", type=int, default=1, help="positive class label (default 1=fall)")
    args = parser.parse_args()

    device = pick_device(args.device)
    print("Using device:", device)

    # Data
    from dataset import get_loaders  # Ensure dataloader.py is in the same directory or PYTHONPATH
    train_loader, val_loader = get_loaders(args.data_dir, batch_size=args.batch_size, normalize=args.normalize)

    model = TinyCNN()  

    # Loss & Optim
    criterion = FocalLoss(alpha=0.2, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    train(model, train_loader, val_loader, criterion, optimizer,
          num_epochs=args.epochs, device=device,
          save_interval=args.save_interval, save_path=args.save_path,
          save_best=args.save_best, eval_interval=args.eval_interval,
          pos_class=args.pos_class)

    # Save final model
    save_model(model, args.save_path)


if __name__ == "__main__":
    main()
