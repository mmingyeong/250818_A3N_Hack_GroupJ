
# CAMELS Cosmology Inference with U-Net

This repository contains experiments using machine learning to infer cosmological parameters (Ωm, σ₈) from the CAMELS multifield dataset.

## 📂 Project Structure
- `data/` : (ignored) raw CAMELS maps & parameter files
- `unet/` : model, training, and evaluation code
- `01_data_analysis.ipynb` : initial data exploration
- `02_unet_debug.ipynb` : debugging and prototype runs
- `03_unet_eval.ipynb` : evaluation and visualization
- `runs/`, `predictions/` : (ignored) training outputs and predictions

## 🚀 Training
Example (SIMBA Mtot):
```bash
python unet/train.py \
  --params_path ./data/CAMELS_multifield/params_LH_SIMBA.txt \
  --imgs_path ./data/CAMELS_multifield/Maps_Mtot_SIMBA_LH_z=0.00.npy \
  --save_dir ./runs/SIMBA_Mtot/ \
  --epochs 50 --batch_size 16 --lr 1e-4 --seed 42 --device cuda
````

## 📊 Evaluation

See `03_unet_eval.ipynb` for:

* Loss curves
* Scatter plots (true vs pred)
* Residual analysis
* Cross-simulation tests (SIMBA ↔ TNG)

## ⚠️ Note

* Large data (`data/`), model checkpoints (`*.pt`), and logs are excluded from the repository.
* To reproduce results, download CAMELS multifield data separately.

---
