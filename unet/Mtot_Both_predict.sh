#!/bin/bash
#PBS -N Mtot_Both_predict
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
LOGFILE="$LOGDIR/Mtot_Both_predict_${PBS_JOBID}.log"
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

# 학습된 모델 체크포인트 경로 (train.py에서 저장한 best_model.pt나 final_model.pt)
MODEL_PATH="/home/users/mmingyeong/250818_a3net/unet/runs/Mtot_Both/20250821_112245/best_model.pt"

# 공통 파라미터
FIELD="Mtot"
OUT_DIM=2              # 훈련 시 설정과 동일해야 함
BATCH_SIZE=32
DEVICE="cuda"

# ---------- 예측 실행 ----------
for SIM in SIMBA IllustrisTNG; do
    echo "🔮 Predicting for simulation: $SIM (field=$FIELD)"
    
    OUTPUT_DIR="$PBS_O_WORKDIR/predictions/${FIELD}_${SIM}"
    mkdir -p "$OUTPUT_DIR"
    
    CMD=( "$PYTHON_BIN" -u "$PREDICT_PY"
          --data_path "$DATA_ROOT"
          --sim "$SIM"
          --field "$FIELD"
          --model_path "$MODEL_PATH"
          --out_dim "$OUT_DIM"
          --output_dir "$OUTPUT_DIR"
          --batch_size "$BATCH_SIZE"
          --device "$DEVICE"
          --save_formats csv hdf5 npy
        )
    
    echo "▶ ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/predict.log"
done

echo "✅ All predictions finished at $(date)"
