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
    - Single mode: one map type -> (N, 15, H, W)
    - Multi mode:
        * concat=True  -> (N, 15*k, H, W)
        * concat=False -> tuple of (N, 15, H, W) maps (for shared encoder + fusion)
    """

    def __init__(self, maps, params, mode: str = "2params",
                 augment: bool = False, multi_map: bool = False, concat: bool = True):
        self.mode = mode
        self.augment = augment
        self.multi_map = multi_map
        self.concat = concat

        # --- maps 로드 ---
        if isinstance(maps, (list, tuple)):  # 여러 맵 파일 or 배열
            loaded_maps = []
            for m in maps:
                if isinstance(m, str):
                    arr = np.load(m)
                    logger.info(f"🗂️ Loaded maps from file: {m}, shape={arr.shape}")
                elif isinstance(m, np.ndarray):
                    arr = np.asarray(m)
                    logger.info(f"🧠 Using in-memory maps: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    raise TypeError(f"maps element must be str or np.ndarray, got {type(m)}")
                loaded_maps.append(arr)

            if self.concat:  # 기존 channel-wise concat
                self.maps = np.concatenate(loaded_maps, axis=1)
                logger.info(f"🔗 Multi-map concat mode: concatenated shape={self.maps.shape}")
            else:  # shared encoder + fusion → 리스트 유지
                self.maps = loaded_maps
                logger.info(f"🔗 Multi-map fusion mode: {[m.shape for m in self.maps]}")

        else:  # 단일 맵
            if isinstance(maps, str):
                self.maps = np.load(maps)  # (N, 15, H, W)
                logger.info(f"🗂️ Loaded maps from file: {maps}")
            elif isinstance(maps, np.ndarray):
                self.maps = np.asarray(maps)
                logger.info(f"🧠 Using in-memory maps: shape={self.maps.shape}, dtype={self.maps.dtype}")
            else:
                raise TypeError(f"`maps` must be str, np.ndarray, or list, got {type(maps)}")

        # --- params 로드 ---
        if isinstance(params, str):
            par = np.loadtxt(params)   # (N, 6) or (6, N)
            logger.info(f"🗂️ Loaded params from file: {params}")
        elif isinstance(params, np.ndarray):
            par = np.asarray(params)
            logger.info(f"🧠 Using in-memory params: shape={par.shape}, dtype={par.dtype}")
        else:
            raise TypeError(f"`params` must be str or np.ndarray, got {type(params)}")

        # (N,6) 보정
        if par.ndim != 2:
            raise ValueError(f"params must be 2D, got shape {par.shape}")

        # 샘플 개수 확인
        if self.concat or not self.multi_map:  # concat or single
            N = self.maps.shape[0]
        else:  # fusion
            N = self.maps[0].shape[0]

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

        # 채널 수 기록
        if self.concat or not self.multi_map:  # concat or single
            self.in_channels = self.maps.shape[1]
        else:  # fusion → 각 맵별 15 채널
            self.in_channels = self.maps[0].shape[1]

        # 최종 확인
        assert N == self.params.shape[0], \
            f"❌ mismatch: maps {N} vs params {self.params.shape[0]}"

        logger.info(
            f"✅ Dataset ready | multi_map={self.multi_map}, concat={self.concat} | "
            f"params[{mode}]: shape={self.params.shape}"
        )

    def __len__(self):
        if self.concat or not self.multi_map:
            return self.maps.shape[0]
        else:
            return self.maps[0].shape[0]

    def __getitem__(self, idx):
        if self.concat or not self.multi_map:
            x = torch.tensor(self.maps[idx], dtype=torch.float32)  # (C, H, W)
        else:  # fusion 모드 → tuple 반환
            x = tuple(torch.tensor(m[idx], dtype=torch.float32) for m in self.maps)

        y = torch.tensor(self.params[idx], dtype=torch.float32)  # (2,) or (6,)

        # --- Data Augmentation ---
        def augment_map(x_map):
            k = random.randint(0, 3)
            x_map = torch.rot90(x_map, k, dims=(-2, -1))
            if random.random() < 0.5:
                x_map = torch.flip(x_map, dims=[-1])
            if random.random() < 0.5:
                x_map = torch.flip(x_map, dims=[-2])
            return x_map

        if self.augment:
            if isinstance(x, tuple):
                x = tuple(augment_map(xi) for xi in x)
            else:
                x = augment_map(x)

        return x, y


# ----------------------------
# Split + DataLoader
# ----------------------------
def get_dataloaders(maps, params_file, batch_size=16, mode="2params",
                    split=(0.6, 0.2, 0.2), num_workers=0, shuffle=True, seed=42,
                    multi_map=False, concat=True):
    """
    Returns train/val/test DataLoaders with reproducible random split.
    `maps` can be str, ndarray, or list of them (for multi_map=True).
    If multi_map=True and concat=False, returns tuple inputs (fusion mode).
    """
    full_dataset = CAMELSDataset(maps, params_file, mode=mode,
                                 augment=False, multi_map=multi_map, concat=concat)
    n_total = len(full_dataset)
    n_train = int(split[0] * n_total)
    n_val   = int(split[1] * n_total)
    n_test  = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )

    train_dataset = Subset(
        CAMELSDataset(maps, params_file, mode=mode, augment=True, multi_map=multi_map, concat=concat),
        train_idx.indices
    )
    val_dataset = Subset(
        CAMELSDataset(maps, params_file, mode=mode, augment=False, multi_map=multi_map, concat=concat),
        val_idx.indices
    )
    test_dataset = Subset(
        CAMELSDataset(maps, params_file, mode=mode, augment=False, multi_map=multi_map, concat=concat),
        test_idx.indices
    )

    logger.info("📊 Dataset split summary:")
    logger.info(f"   total samples = {n_total}")
    logger.info(f"   split ratio   = {split}")
    logger.info(f"   train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}")
    logger.info(f"   batch size    = {batch_size}, num_workers = {num_workers}, shuffle = {shuffle}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, full_dataset.in_channels
