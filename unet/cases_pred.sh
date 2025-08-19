#!/bin/bash
#PBS -N predict_camels_cases
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M mmingyeong@kasi.re.kr

set -eo pipefail
cd "$PBS_O_WORKDIR"

LOGDIR="$PBS_O_WORKDIR/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/predict_camels_cases_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"
echo "📂 Working Dir: $PBS_O_WORKDIR"
echo "🐍 Python Path (pre-activate): $(which python || true)"
echo "🧪 Python Version (pre-activate): $(python --version || true)"
nvidia-smi || echo "⚠️ No GPU detected or nvidia-smi not available"

# --- conda init ---
if [ -f /opt/anaconda3/2023.03/etc/profile.d/conda.sh ]; then
  source /opt/anaconda3/2023.03/etc/profile.d/conda.sh
else
  [ -f ~/.bashrc ] && source ~/.bashrc || true
fi
conda activate py312

echo "🐍 Python Path (post-activate): $(which python || true)"
echo "🧪 Python Version (post-activate): $(python --version || true)"
echo "🚀 Starting prediction job on $(hostname) at $(date)"

# -----------------------------
# 기본 설정
# -----------------------------
PYTHON_BIN="${PYTHON:-python}"
PREDICT_PY="/home/users/mmingyeong/250818_a3net/unet/predict.py"
DATA_ROOT="/caefs/user/mmingyeong/250818_a3net/data/CAMELS_multifield"
RUNS_ROOT="${RUNS_ROOT:-$PBS_O_WORKDIR/runs}"      # runs/case_#/.../<ts>/best_model.pt
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"

# -----------------------------
# 입력 4종 (단일 맵)
# -----------------------------
declare -A INPUTS
INPUTS["SIMBA_Mtot"]="SIMBA Mtot"
INPUTS["TNG_Mtot"]="IllustrisTNG Mtot"
INPUTS["SIMBA_P"]="SIMBA P"
INPUTS["TNG_P"]="IllustrisTNG P"

# -----------------------------
# 출력 3종 (모두 out_dim=2)
# -----------------------------
declare -A OUTPUTS
OUTPUTS["A"]="--out_dim 2 --param_names Omega_m,sigma_8"
OUTPUTS["B"]="--out_dim 2 --param_names AGN1,AGN2"
OUTPUTS["C"]="--out_dim 2 --param_names SN1,SN2"

# -----------------------------
# 실행 루프 (총 12 케이스)
# -----------------------------
for INPUT_KEY in SIMBA_Mtot TNG_Mtot SIMBA_P TNG_P; do
  read -r SIM FIELD <<< "${INPUTS[$INPUT_KEY]}"

  for OUT_KEY in A B C; do
    # 최신 best_model.pt 찾기 (따옴표 없이 글롭 확장)
    MODEL_GLOB=${RUNS_ROOT}/case_*_${INPUT_KEY}_${OUT_KEY}/*/best_model.pt
    MODEL_FILE=$(ls -t $MODEL_GLOB 2>/dev/null | head -n 1 || true)

    if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
      echo "❌ No best_model.pt for ${INPUT_KEY}/${OUT_KEY} (pattern: $MODEL_GLOB)"
      continue
    fi

    # runs 포맷 그대로 베이스/타임스탬프 추출
    RUN_TS_DIR="$(basename "$(dirname "$MODEL_FILE")")"         # <ts> (공백/언더스코어 그대로 유지)
    CASE_BASE="$(basename "$(dirname "$(dirname "$MODEL_FILE")")")"  # case_#_INPUT_OUTKEY

    SAVE_PARENT="$PBS_O_WORKDIR/predictions/${CASE_BASE}"
    SAVE_DIR="${SAVE_PARENT}/${RUN_TS_DIR}"
    mkdir -p "$SAVE_DIR"

    echo "---------------------------------------------"
    echo "🔮 ${CASE_BASE}"
    echo "    SIM/Field : $SIM / $FIELD"
    echo "    Model     : $MODEL_FILE"
    echo "    Save      : $SAVE_DIR"
    echo "---------------------------------------------"

    CMD=( "$PYTHON_BIN" -u "$PREDICT_PY"
          --data_path "$DATA_ROOT"
          --sim "$SIM"
          --field "$FIELD"
          --model_path "$MODEL_FILE"
          --output_dir "$SAVE_DIR"
          --batch_size "$BATCH_SIZE"
          --device "$DEVICE"
        )
    # shellcheck disable=SC2206
    OUT_ARGS=(${OUTPUTS[$OUT_KEY]})
    CMD+=( "${OUT_ARGS[@]}" )

    echo "▶ ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "$SAVE_DIR/predict.log"

    echo "✅ Finished ${CASE_BASE} at $(date)"
  done
done

echo "🎉 All predictions done at $(date)"
