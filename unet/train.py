
import argparse
import os
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import UNetEncoderRegressor
from data_loader import CAMELSDataset


# ---------------------------
# Early Stopping 클래스
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ---------------------------
# Logging 설정
# ---------------------------
def setup_logger(log_file=None):
    import logging, sys, os
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True
    )
    return logging.getLogger("train")


# ---------------------------
# Training 함수
# ---------------------------
def train(args):
    logger = setup_logger(os.path.join(args.save_dir, "train.log"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("🚀 Starting training with configuration:")
    for k, v in vars(args).items():
        logger.info(f"   {k}: {v}")

    # params concat
    params_paths = args.params_path.split(",")
    params = np.concatenate([np.loadtxt(p) for p in params_paths], axis=0)

    # imgs concat
    imgs_paths = args.imgs_path.split(",")
    imgs_list = []
    for p in imgs_paths:
        arr = np.load(p)
        if arr.ndim == 3 and arr.shape[0] % 15 == 0:
            n_sims = arr.shape[0] // 15
            arr = arr.reshape(n_sims, 15, arr.shape[1], arr.shape[2])
        imgs_list.append(arr.astype(np.float32, copy=False))

    imgs = np.concatenate(imgs_list, axis=0)

    # 검증
    if imgs.shape[0] != params.shape[0]:
        raise ValueError(f"❌ Count mismatch: imgs={imgs.shape[0]} vs params={params.shape[0]}")

    logger.info(f"📂 Final dataset shape: imgs={imgs.shape}, params={params.shape}")



    # Train/Val/Test split index
    n_total = len(params)
    train_idx, test_idx = train_test_split(np.arange(n_total), test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=args.seed)

    def make_subset(idxs):
        param_mode = "2params" if args.out_dim == 2 else "6params"
        return CAMELSDataset(imgs[idxs], params[idxs], mode=param_mode)


    train_dataset = make_subset(train_idx)
    val_dataset   = make_subset(val_idx)
    test_dataset  = make_subset(test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"📊 Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # 모델 초기화
    # 모델 초기화
    model = UNetEncoderRegressor(
        in_channels=15,
        out_dim=args.out_dim,   # ✅ argparse 값 사용
        mode=args.mode,
        fusion=args.fusion
    ).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_records = []
    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # 학습 루프
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training", leave=False)
        for X, y in pbar:
            if args.mode == "single":
                X = X.to(args.device)
            else:
                # multi → list of tensors
                X = [m.to(args.device) for m in X]
            y = y.to(args.device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation", leave=False)
            for X, y in pbar_val:
                if args.mode == "single":
                    X = X.to(args.device)
                else:
                    X = [m.to(args.device) for m in X]
                y = y.to(args.device)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item()
                pbar_val.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        log_records.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✅ Epoch {epoch+1}: Best model updated (val_loss={avg_val_loss:.6f})")

        logger.info(
            f"📉 Epoch {epoch+1:03d}/{args.epochs} "
            f"| Train Loss: {avg_train_loss:.6f} "
            f"| Val Loss: {avg_val_loss:.6f} "
            f"| LR: {current_lr:.2e}"
        )

        if early_stopper(avg_val_loss):
            logger.info(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    # Final model 저장
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"📦 Final model saved: {final_model_path}")

    # 로그 CSV 저장
    csv_path = os.path.join(args.save_dir, "training_log.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(log_records)
    logger.info(f"📝 Training log saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNetEncoderRegressor on CAMELS maps")
    parser.add_argument("--params_path", type=str, required=True, help="Path to params file")
    parser.add_argument("--imgs_path", type=str, required=True,
                        help="Path(s) to images file(s). If multi mode, use comma-separated list")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models and logs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement in val_loss to reset patience")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"],
                        help="Experiment mode: single or multi")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "mean", "sum"],
                        help="Fusion method for multi mode")

    # ✅ 새 인자 추가
    parser.add_argument("--out_dim", type=int, default=2, choices=[2, 6],
                        help="Number of cosmological parameters to predict")
    parser.add_argument("--param_names", type=str, default=None,
                        help="Comma-separated list of parameter names (must match out_dim)")

    args = parser.parse_args()
    train(args)
