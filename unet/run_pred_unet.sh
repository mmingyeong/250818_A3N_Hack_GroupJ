#!/bin/bash
#PBS -N predict_camels_unetenc
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M mmingyeong@kasi.re.kr

cd "$PBS_O_WORKDIR"

LOGDIR="$PBS_O_WORKDIR/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/predict_camels_unetenc_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"
echo "📂 Working Dir: $PBS_O_WORKDIR"
echo "🐍 Python Path: $(which python || true)"
echo "🧪 Python Version: $(python --version || true)"
nvidia-smi || echo "⚠️ No GPU detected or nvidia-smi not available"

source ~/.bashrc
conda activate py312

echo "🚀 Starting prediction job on $(hostname) at $(date)"

# ---------- 사용자 설정 ----------
PYTHON_BIN="${PYTHON:-python}"
PREDICT_PY="/home/users/mmingyeong/250818_a3net/unet/predict.py"
DATA_ROOT="/caefs/user/mmingyeong/250818_a3net/data/CAMELS_multifield"
MODEL_PATH="/home/users/mmingyeong/250818_a3net/unet/runs/SIMBA_Mtot/20250819_003807/best_model.pt"

: "${EXP_LIST:=SIMBA_Mtot}"   # qsub -v EXP_LIST="SIMBA_Mtot TNG_Mtot" 로 여러 실험 지정 가능
OUT_DIM=2
BATCH_SIZE=16
SEED=42
DEVICE=cuda

resolve_paths () {
  local EXP_NAME="$1"
  case "$EXP_NAME" in
    SIMBA_Mtot)
      SIM="SIMBA"
      FIELDS="Mtot"
      ;;
    SIMBA_P)
      SIM="SIMBA"
      FIELDS="P"
      ;;
    TNG_Mtot|IllustrisTNG_Mtot)
      SIM="IllustrisTNG"
      FIELDS="Mtot"
      ;;
    TNG_P|IllustrisTNG_P)
      SIM="IllustrisTNG"
      FIELDS="P"
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
  SAVE_DIR="$PBS_O_WORKDIR/predictions/${EXP}/${TS}"
  mkdir -p "$SAVE_DIR"

  echo "---------------------------------------------"
  echo "🔮 Prediction Experiment: $EXP"
  echo "    Data Root : $DATA_ROOT"
  echo "    Sim       : $SIM"
  echo "    Fields    : $FIELDS"
  echo "    Model     : $MODEL_PATH"
  echo "    Save Dir  : $SAVE_DIR"
  echo "    Predict   : $PREDICT_PY"
  echo "---------------------------------------------"

  CMD=( "$PYTHON_BIN" -u "$PREDICT_PY"
        --data_path "$DATA_ROOT"
        --sim "$SIM"
        --fields "$FIELDS"
        --out_dim "$OUT_DIM"
        --model_path "$MODEL_PATH"
        --output_dir "$SAVE_DIR"
        --batch_size "$BATCH_SIZE"
        --seed "$SEED"
        --device "$DEVICE"
      )

  echo "▶ ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "$SAVE_DIR/predict.log"

  echo "✅ Finished $EXP at $(date)"
done

echo "🎉 All predictions done at $(date)"