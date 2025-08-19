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

    # Seed 고정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("🚀 Starting training with configuration:")
    for k, v in vars(args).items():
        logger.info(f"   {k}: {v}")

    # --- 데이터 로드 ---
    params = np.loadtxt(args.params_path)   # (1000, 6) 혹은 (6, 1000)
    imgs = np.load(args.imgs_path)          # (15000, 256, 256) 또는 (1000, 15, 256, 256)

    # params shape 보정 (파일에 따라 (6, N)인 경우 존재)
    if params.ndim != 2:
        raise ValueError(f"params must be 2D, got {params.shape}")
    if params.shape[0] != imgs.shape[0] and params.shape[1] == imgs.shape[0]:
        params = params.T

    # ---- 채널 축 복원 로직 ----
    # Case A: (N, 15, 256, 256) 이미 정상
    if imgs.ndim == 4 and imgs.shape[1] == 15:
        logger.info(f"🟢 Detected shape with channels: imgs={imgs.shape}")
    # Case B: (15000, 256, 256) 처럼 채널 축이 빠진 경우 → (1000, 15, 256, 256)로 복원
    elif imgs.ndim == 3 and imgs.shape[0] % 15 == 0:
        n_sims = imgs.shape[0] // 15
        imgs = imgs.reshape(n_sims, 15, imgs.shape[1], imgs.shape[2])
        logger.info(f"🔧 Reshaped imgs to {imgs.shape} (assumed 15 maps per simulation)")
    else:
        raise ValueError(f"❌ Unexpected imgs shape: {imgs.shape}. "
                        f"Expected (N,15,H,W) or (N*15,H,W).")

    # 이제 imgs.shape[0] == params.shape[0] 여야 함
    if params.shape[0] != imgs.shape[0]:
        raise ValueError(f"❌ Count mismatch after reshape: imgs={imgs.shape[0]} vs params={params.shape[0]}")

    # (선택) dtype 정리
    imgs = imgs.astype(np.float32, copy=False)
    params = params.astype(np.float32, copy=False)

    logger.info(f"📂 Final dataset view: imgs={imgs.shape}, params={params.shape}")


    # Train/Validation/Test split
    train_idx, test_idx = train_test_split(np.arange(len(imgs)), test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=args.seed)  # 0.25*0.8=0.2

    train_dataset = CAMELSDataset(imgs[train_idx], params[train_idx])
    val_dataset = CAMELSDataset(imgs[val_idx], params[val_idx])
    test_dataset = CAMELSDataset(imgs[test_idx], params[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"📊 Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # 모델 초기화
    model = UNetEncoderRegressor(in_channels=15, out_dim=2).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_records = []

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    final_model_path = os.path.join(args.save_dir, "final_model.pt")

    # 학습 루프
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training", leave=False)
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(args.device), y.to(args.device)
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
                X, y = X.to(args.device), y.to(args.device)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item()
                pbar_val.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        # 현재 LR 가져오기
        current_lr = optimizer.param_groups[0]["lr"]

        # 로그 기록
        log_records.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })

        # Best model 저장
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
    parser.add_argument("--imgs_path", type=str, required=True, help="Path to images file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models and logs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train(args)
