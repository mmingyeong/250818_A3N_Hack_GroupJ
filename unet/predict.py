"""
unet/predict.py

Run inference using a trained UNetEncoderRegressor model and save predictions
alongside the true cosmological parameters (CSV + HDF5), supporting out_dim=2 or 6.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-08-18
"""

import os
import argparse
import logging
import torch
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import UNetEncoderRegressor
from data_loader import CAMELSDataset


# ---------------------------
# Logging setup
# ---------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def resolve_param_names(out_dim: int, custom_names: str | None = None):
    """
    Decide column names for parameters based on out_dim or a user-specified list.

    custom_names: comma-separated string like "Omega_m,sigma_8,SN1,AGN1,SN2,AGN2"
    """
    if custom_names:
        names = [s.strip() for s in custom_names.split(",") if s.strip()]
        if len(names) != out_dim:
            raise ValueError(
                f"--param_names length ({len(names)}) must match --out_dim ({out_dim})."
            )
        return names

    if out_dim == 2:
        return ["Omega_m", "sigma_8"]
    if out_dim == 6:
        return ["Omega_m", "sigma_8", "SN1", "AGN1", "SN2", "AGN2"]
    # fallback: generic names
    return [f"param_{i}" for i in range(out_dim)]


# ---------------------------
# Prediction function
# ---------------------------
def run_prediction(args):
    setup_logger()

    param_names = resolve_param_names(args.out_dim, args.param_names)
    logging.info(f"🧩 Parameter names: {param_names}")

    logging.info(f"📦 Loading model from: {args.model_path}")
    model = UNetEncoderRegressor(in_channels=15, out_dim=args.out_dim).to(args.device)
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    maps = np.load(os.path.join(args.data_path, f"Maps_{args.fields}_{args.sim}_LH_z=0.00.npy"))
    params = np.loadtxt(os.path.join(args.data_path, f"params_LH_{args.sim}.txt"))

    # train.py와 동일한 reshape
    n_params = params.shape[0]
    maps = maps.reshape(n_params, 15, 256, 256)  # ✅ 핵심!

    dataset = CAMELSDataset(maps, params, mode="2params" if args.out_dim==2 else "6params")


    indices = list(range(len(dataset)))
    # 8:1:1 split (train:val:test) with fixed seed
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125 / 0.85, random_state=args.seed)  # 0.125 = 1/8

    test_set = Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    logging.info(
        f"🧪 Inference split sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )
    logging.info(f"🧪 Running inference on {len(test_set)} test samples")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="🚀 Predicting"):
            x, y = x.to(args.device), y.to(args.device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Sanity check
    if all_preds.shape[1] != args.out_dim or all_targets.shape[1] != args.out_dim:
        raise RuntimeError(
            f"Shape mismatch: preds {all_preds.shape}, targets {all_targets.shape}, out_dim={args.out_dim}"
        )

    logging.info(f"📐 Prediction complete | y_pred shape: {all_preds.shape} | y_true shape: {all_targets.shape}")

    # Save predictions (HDF5 + CSV)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- HDF5 ----
    output_h5 = os.path.join(args.output_dir, "predictions.h5")
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("y_true", data=all_targets)
        f.create_dataset("y_pred", data=all_preds)
        f.attrs["sim"] = args.sim
        f.attrs["fields"] = args.fields if args.fields else "Mtot"
        f.attrs["out_dim"] = args.out_dim
        f.attrs["param_names"] = ",".join(param_names)
        f.attrs["seed"] = args.seed
    logging.info(f"✅ Predictions saved to {output_h5}")

    # ---- CSV ----
    # Build dict for CSV columns: true_*, pred_*
    data = {}
    for i, name in enumerate(param_names):
        data[f"true_{name}"] = all_targets[:, i]
    for i, name in enumerate(param_names):
        data[f"pred_{name}"] = all_preds[:, i]

    # Include metadata columns for convenience
    data["sim"] = [args.sim] * all_preds.shape[0]
    data["fields"] = [args.fields if args.fields else "Mtot"] * all_preds.shape[0]
    data["seed"] = [args.seed] * all_preds.shape[0]

    df = pd.DataFrame(data)
    output_csv = os.path.join(args.output_dir, "predictions.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"✅ Predictions also saved to {output_csv}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNetEncoderRegressor inference on CAMELS test set (CSV/HDF5)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CAMELS dataset root (npy/txt)")
    parser.add_argument("--sim", type=str, default="SIMBA", choices=["IllustrisTNG", "SIMBA"])
    parser.add_argument("--fields", type=str, default="Mtot", help="Comma-separated fields, e.g., 'Mtot,P'")
    parser.add_argument("--out_dim", type=int, default=2, choices=[2, 6], help="Number of output parameters")
    parser.add_argument("--param_names", type=str, default=None,
                        help="Optional comma-separated names for parameters (must match out_dim)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    run_prediction(args)