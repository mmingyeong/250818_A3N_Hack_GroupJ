#!/bin/bash
#PBS -N Mtot_Both_train
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M mmingyeong@kasi.re.kr

cd "$PBS_O_WORKDIR"

LOGDIR="$PBS_O_WORKDIR/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/Mtot_Both_train_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"
echo "📂 Working Dir: $PBS_O_WORKDIR"
echo "🐍 Python Path: $(which python || true)"
echo "🧪 Python Version: $(python --version || true)"
nvidia-smi || echo "⚠️ No GPU detected or nvidia-smi not available"

source ~/.bashrc
conda activate py312

echo "🚀 Starting training job on $(hostname) at $(date)"

# ---------- 사용자 설정 ----------
PYTHON_BIN="${PYTHON:-python}"
TRAIN_PY="/home/users/mmingyeong/250818_a3net/unet/train.py"
DATA_ROOT="/caefs/user/mmingyeong/250818_a3net/data/CAMELS_multifield"

PARAMS_PATH="$DATA_ROOT/params_LH_SIMBA.txt,$DATA_ROOT/params_LH_IllustrisTNG.txt"
IMGS_PATH="$DATA_ROOT/Maps_Mtot_SIMBA_LH_z=0.00.npy,$DATA_ROOT/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"

# 기본 하이퍼파라미터
SEED=42
EPOCHS=30
BATCH_SIZE=16
LR=1e-4
DEVICE=cuda

# 저장 경로
TS="$(date +"%Y%m%d_%H%M%S")"
SAVE_DIR="$PBS_O_WORKDIR/runs/Mtot_Both/${TS}"
mkdir -p "$SAVE_DIR"

echo "---------------------------------------------"
echo "🚀 Experiment: SIMBA+TNG Mtot (concatenated)"
echo "    Params: $PARAMS_PATH"
echo "    Images: $IMGS_PATH"
echo "    Save  : $SAVE_DIR"
echo "    Train : $TRAIN_PY"
echo "---------------------------------------------"

CMD=( "$PYTHON_BIN" -u "$TRAIN_PY"
      --params_path "$PARAMS_PATH"
      --imgs_path "$IMGS_PATH"
      --save_dir "$SAVE_DIR"
      --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --lr "$LR"
      --seed "$SEED"
      --device "$DEVICE"
    )

echo "▶ ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$SAVE_DIR/train.log"

echo "✅ Finished at $(date)"
echo "🎉 All done!"
