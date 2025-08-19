#!/bin/bash
#PBS -N train_camels_unetenc
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
LOGFILE="$LOGDIR/train_camels_unetenc_${PBS_JOBID}.log"
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
# ✅ 실제 train.py 경로로 수정
TRAIN_PY="/home/users/mmingyeong/250818_a3net/unet/train.py"
DATA_ROOT="/caefs/user/mmingyeong/250818_a3net/data/CAMELS_multifield"

: "${EXP_LIST:=SIMBA_Mtot}"   # qsub -v EXP_LIST="SIMBA_Mtot TNG_Mtot" 로 덮어쓰기 가능
SEED=42
EPOCHS=50
BATCH_SIZE=16
LR=1e-4
DEVICE=cuda

resolve_paths () {
  local EXP_NAME="$1"
  case "$EXP_NAME" in
    SIMBA_Mtot)
      PARAMS_PATH="$DATA_ROOT/params_LH_SIMBA.txt"
      IMGS_PATH="$DATA_ROOT/Maps_Mtot_SIMBA_LH_z=0.00.npy"
      ;;
    SIMBA_P)
      PARAMS_PATH="$DATA_ROOT/params_LH_SIMBA.txt"
      IMGS_PATH="$DATA_ROOT/Maps_P_SIMBA_LH_z=0.00.npy"
      ;;
    TNG_Mtot|IllustrisTNG_Mtot)
      PARAMS_PATH="$DATA_ROOT/params_LH_IllustrisTNG.txt"
      IMGS_PATH="$DATA_ROOT/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"
      ;;
    TNG_P|IllustrisTNG_P)
      PARAMS_PATH="$DATA_ROOT/params_LH_IllustrisTNG.txt"
      IMGS_PATH="$DATA_ROOT/Maps_P_IllustrisTNG_LH_z=0.00.npy"
      ;;
    *)
      echo "❌ Unknown experiment name: $EXP_NAME"
      return 1
      ;;
  esac
  return 0
}

for EXP in $EXP_LIST; do
  resolve_paths "$EXP" || { echo "Skip $EXP"; continue; }

  TS="$(date +"%Y%m%d_%H%M%S")"
  SAVE_DIR="$PBS_O_WORKDIR/runs/${EXP}/${TS}"
  mkdir -p "$SAVE_DIR"

  echo "---------------------------------------------"
  echo "🚀 Experiment: $EXP"
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
        --patience 20 
        --min_delta 1e-3
      )

  echo "▶ ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "$SAVE_DIR/train.log"

  echo "✅ Finished $EXP at $(date)"
done

echo "🎉 All done at $(date)"