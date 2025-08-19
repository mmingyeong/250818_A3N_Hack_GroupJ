
import os
import argparse
import logging
import torch
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNetEncoderRegressor
from data_loader import CAMELSDataset


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def resolve_param_names(out_dim: int, custom_names: str | None = None):
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
    return [f"param_{i}" for i in range(out_dim)]


def run_prediction(args):
    setup_logger()

    param_names = resolve_param_names(args.out_dim, args.param_names)
    logging.info(f"🧩 Parameter names: {param_names}")

    mode = "2params" if args.out_dim == 2 else "6params"

    # --- maps 로드 ---
    map_file = os.path.join(args.data_path, f"Maps_{args.field}_{args.sim}_LH_z=0.00.npy")
    maps = np.load(map_file)

    # ✅ shape 보정: (15000, 256, 256) → (1000, 15, 256, 256)
    if maps.ndim == 3 and maps.shape[0] % 15 == 0:
        n_sims = maps.shape[0] // 15
        maps = maps.reshape(n_sims, 15, maps.shape[1], maps.shape[2])
        logging.info(f"🔄 Reshaped maps to {maps.shape}")
    else:
        logging.info(f"📂 Loaded maps with shape {maps.shape}")

    # --- params 로드 ---
    params_file = os.path.join(args.data_path, f"params_LH_{args.sim}.txt")
    params = np.loadtxt(params_file)

    if params.ndim != 2:
        raise ValueError(f"params must be 2D, got {params.shape}")
    if params.shape[0] != maps.shape[0] and params.shape[1] == maps.shape[0]:
        params = params.T

    if params.shape[0] != maps.shape[0]:
        raise ValueError(f"❌ params count {params.shape[0]} does not match maps N={maps.shape[0]}")

    dataset = CAMELSDataset(
        maps,
        params,
        mode=mode,
        augment=False,
        multi_map=False
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    in_channels = dataset.in_channels
    model = UNetEncoderRegressor(
        in_channels=in_channels,
        out_dim=args.out_dim,
        mode="single"
    ).to(args.device)

    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    # --- Prediction ---
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="🚀 Predicting"):
            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)
            preds.append(y_pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    logging.info(f"📐 Prediction complete | y_pred shape: {preds.shape} | y_true shape: {trues.shape}")

    # --- 저장 ---
    os.makedirs(args.output_dir, exist_ok=True)

    if "hdf5" in args.save_formats:
        output_h5 = os.path.join(args.output_dir, "predictions.h5")
        with h5py.File(output_h5, "w") as f:
            f.create_dataset("y_true", data=trues)
            f.create_dataset("y_pred", data=preds)
            f.attrs["sim"] = args.sim
            f.attrs["field"] = args.field
            f.attrs["out_dim"] = args.out_dim
            f.attrs["param_names"] = ",".join(param_names)
        logging.info(f"✅ Predictions saved to {output_h5}")

    if "csv" in args.save_formats:
        data = {}
        for i, name in enumerate(param_names):
            data[f"true_{name}"] = trues[:, i]
            data[f"pred_{name}"] = preds[:, i]
        df = pd.DataFrame(data)
        output_csv = os.path.join(args.output_dir, "predictions.csv")
        df.to_csv(output_csv, index=False)
        logging.info(f"✅ Predictions also saved to {output_csv}")

    if "npy" in args.save_formats:
        np.save(os.path.join(args.output_dir, "preds.npy"), preds)
        np.save(os.path.join(args.output_dir, "trues.npy"), trues)
        logging.info("✅ Predictions saved as .npy files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sim", type=str, default="IllustrisTNG",
                        choices=["IllustrisTNG", "SIMBA"])
    parser.add_argument("--field", type=str, default="Mtot",
                        help="Single field, e.g., 'Mtot' or 'P'")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dim", type=int, default=2, choices=[2, 6])
    parser.add_argument("--param_names", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_formats", type=str, nargs="+", default=["csv", "hdf5", "npy"])
    args = parser.parse_args()

    run_prediction(args)
