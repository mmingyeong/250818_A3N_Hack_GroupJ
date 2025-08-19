# unet/data_loader.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger("data_loader")


class CAMELSDataset(Dataset):
    """
    CAMELS 2D maps + params dataset.
    - maps: (N, 15, 256, 256)
    - params: (N, 6)  (mode='2params'이면 앞의 2개만 사용)
    """

    def __init__(self, maps, params, mode: str = "2params", augment: bool = False):
        # --- maps 로드: path or ndarray ---
        if isinstance(maps, str):
            self.maps = np.load(maps)  # (N, 15, 256, 256)
            logger.info(f"🗂️ Loaded maps from file: {maps}")
        elif isinstance(maps, np.ndarray):
            self.maps = np.asarray(maps)
            logger.info(f"🧠 Using in-memory maps: shape={self.maps.shape}, dtype={self.maps.dtype}")
        else:
            raise TypeError(f"`maps` must be str or np.ndarray, got {type(maps)}")

        # --- params 로드: path or ndarray ---
        if isinstance(params, str):
            par = np.loadtxt(params)   # (N, 6) 또는 (6, N)
            logger.info(f"🗂️ Loaded params from file: {params}")
        elif isinstance(params, np.ndarray):
            par = np.asarray(params)
            logger.info(f"🧠 Using in-memory params: shape={par.shape}, dtype={par.dtype}")
        else:
            raise TypeError(f"`params` must be str or np.ndarray, got {type(params)}")

        # (N,6) 보정
        if par.ndim != 2:
            raise ValueError(f"params must be 2D, got shape {par.shape}")

        N = self.maps.shape[0]
        if par.shape[0] == N:
            pass
        elif par.shape[1] == N:
            par = par.T
            logger.info("🔁 Transposed params to shape (N, C)")
        else:
            raise ValueError(f"❌ params count {par.shape} does not match maps N={N}")

        # 모드 적용
        if mode == "2params":
            self.params = par[:, :2]
        elif mode == "6params":
            self.params = par[:, :6]
        else:
            raise ValueError("mode must be '2params' or '6params'")

        assert self.maps.shape[0] == self.params.shape[0], \
            f"❌ mismatch: maps {self.maps.shape[0]} vs params {self.params.shape[0]}"

        self.mode = mode
        self.augment = augment  # augmentation 여부 (train만 True로 설정)

        logger.info(
            "✅ Dataset ready | "
            f"maps: shape={self.maps.shape}, dtype={self.maps.dtype}, "
            f"min={self.maps.min():.3f}, max={self.maps.max():.3f} | "
            f"params[{mode}]: shape={self.params.shape}, dtype={self.params.dtype}"
        )

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.maps[idx], dtype=torch.float32)    # (15, 256, 256)
        y = torch.tensor(self.params[idx], dtype=torch.float32)  # (2,) or (6,)

        # --- Data Augmentation (train만 적용) ---
        if self.augment:
            # 랜덤 회전 (k * 90도)
            k = random.randint(0, 3)
            x = torch.rot90(x, k, dims=(-2, -1))

            # 랜덤 flip
            if random.random() < 0.5:  # horizontal flip
                x = torch.flip(x, dims=[-1])
            if random.random() < 0.5:  # vertical flip
                x = torch.flip(x, dims=[-2])

        return x, y


# ----------------------------
# Split + DataLoader
# ----------------------------
def get_dataloaders(maps_file, params_file, batch_size=16, mode="2params",
                    split=(0.6, 0.2, 0.2), num_workers=0, shuffle=True, seed=42):
    """
    Returns train/val/test DataLoaders with reproducible random split.
    """
    full_dataset = CAMELSDataset(maps_file, params_file, mode=mode, augment=False)
    n_total = len(full_dataset)
    n_train = int(split[0] * n_total)
    n_val   = int(split[1] * n_total)
    n_test  = n_total - n_train - n_val

    # reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )

    # 각 split에 대해 augment 설정 다르게 적용
    train_dataset = Subset(CAMELSDataset(maps_file, params_file, mode=mode, augment=True), train_idx.indices)
    val_dataset   = Subset(CAMELSDataset(maps_file, params_file, mode=mode, augment=False), val_idx.indices)
    test_dataset  = Subset(CAMELSDataset(maps_file, params_file, mode=mode, augment=False), test_idx.indices)

    logger.info("📊 Dataset split summary:")
    logger.info(f"   total samples = {n_total}")
    logger.info(f"   split ratio   = {split}")
    logger.info(f"   train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}")
    logger.info(f"   batch size    = {batch_size}, num_workers = {num_workers}, shuffle = {shuffle}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
