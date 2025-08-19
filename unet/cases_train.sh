#!/bin/bash
#PBS -N train_camels_cases
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M mmingyeong@kasi.re.kr

# 안전 설정: -u 제거 (PS1 nounset 이슈 회피)
set -eo pipefail

cd "$PBS_O_WORKDIR"

LOGDIR="$PBS_O_WORKDIR/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/train_camels_cases_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"
echo "📂 Working Dir: $PBS_O_WORKDIR"
echo "🐍 Python Path (pre-activate): $(which python || true)"
echo "🧪 Python Version (pre-activate): $(python --version || true)"
nvidia-smi || echo "⚠️ No GPU detected or nvidia-smi not available"

# ---- Conda 초기화 (직접) ----
if [ -f /opt/anaconda3/2023.03/etc/profile.d/conda.sh ]; then
  source /opt/anaconda3/2023.03/etc/profile.d/conda.sh
else
  # fallback: conda.sh 경로가 다르면 필요에 따라 수정
  echo "⚠️ conda.sh not found at default path; trying ~/.bashrc as fallback"
  [ -f ~/.bashrc ] && source ~/.bashrc || true
fi
conda activate py312

echo "🐍 Python Path (post-activate): $(which python || true)"
echo "🧪 Python Version (post-activate): $(python --version || true)"
echo "🚀 Starting training job on $(hostname) at $(date)"

# -----------------------------
# 기본 설정
# -----------------------------
PYTHON_BIN="${PYTHON:-python}"
TRAIN_PY="/home/users/mmingyeong/250818_a3net/unet/train.py"   # single-mode train.py
DATA_ROOT="/caefs/user/mmingyeong/250818_a3net/data/CAMELS_multifield"

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-5}"         # 테스트: 5 epoch
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-cuda}"

# -----------------------------
# 입력 4종 (단일 맵만)
# -----------------------------
declare -A INPUTS
INPUTS["SIMBA_Mtot"]="$DATA_ROOT/Maps_Mtot_SIMBA_LH_z=0.00.npy $DATA_ROOT/params_LH_SIMBA.txt"
INPUTS["TNG_Mtot"]="$DATA_ROOT/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy $DATA_ROOT/params_LH_IllustrisTNG.txt"
INPUTS["SIMBA_P"]="$DATA_ROOT/Maps_P_SIMBA_LH_z=0.00.npy $DATA_ROOT/params_LH_SIMBA.txt"
INPUTS["TNG_P"]="$DATA_ROOT/Maps_P_IllustrisTNG_LH_z=0.00.npy $DATA_ROOT/params_LH_IllustrisTNG.txt"

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
CASE_NUM=1
for INPUT_KEY in SIMBA_Mtot TNG_Mtot SIMBA_P TNG_P; do
  read -r IMGS_PATH PARAMS_PATH <<< "${INPUTS[$INPUT_KEY]}"

  for OUT_KEY in A B C; do
    TS="$(date +"%Y%m%d_%H%M%S")"
    SAVE_PARENT="$PBS_O_WORKDIR/runs/case_${CASE_NUM}_${INPUT_KEY}_${OUT_KEY}"
    SAVE_DIR="${SAVE_PARENT}/${TS}"
    mkdir -p "$SAVE_DIR"

    echo "---------------------------------------------"
    echo "🧪 Case $CASE_NUM / 12"
    echo "    Input : $INPUT_KEY"
    echo "    Output: $OUT_KEY  (${OUTPUTS[$OUT_KEY]})"
    echo "    Images: $IMGS_PATH"
    echo "    Params: $PARAMS_PATH"
    echo "    Save  : $SAVE_DIR"
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
          --patience 3
          --min_delta 1e-3
        )

    # --out_dim / --param_names 추가
    # shellcheck disable=SC2206
    OUT_ARGS=(${OUTPUTS[$OUT_KEY]})
    CMD+=( "${OUT_ARGS[@]}" )

    echo "▶ ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "$SAVE_DIR/train.log"

    echo "✅ Finished Case $CASE_NUM at $(date)"
    CASE_NUM=$((CASE_NUM+1))
  done
done

echo "🎉 All 12 cases done at $(date)"
