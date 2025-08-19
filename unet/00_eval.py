import os
import glob
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# =============================
# 설정
# =============================
RUNS_DIR = "./runs"
PREDS_DIR = "./predictions"
SAVE_DIR = "./eval_results"
os.makedirs(SAVE_DIR, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================
# 유틸 함수
# =============================
def load_training_log(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_predictions(pred_dir):
    """pred_dir: 타임스탬프 하위 폴더 경로"""
    true_path = os.path.join(pred_dir, "trues.npy")
    pred_path = os.path.join(pred_dir, "preds.npy")
    if not (os.path.exists(true_path) and os.path.exists(pred_path)):
        return None
    y_true = np.load(true_path)
    y_pred = np.load(pred_path)
    return y_true, y_pred

def compute_metrics(y_true, y_pred, param_names):
    EPS = 1e-8
    rows = []
    for i, pname in enumerate(param_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        resid = yp - yt
        rows.append({
            "param": pname,
            "R2": r2_score(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE": mean_absolute_error(yt, yp),
            "Pearson_r": pearsonr(yt, yp)[0],
            "Spearman_rho": spearmanr(yt, yp)[0],
            "Bias": np.mean(resid),
            "Std": np.std(resid, ddof=1),
            "RelErr": 100.0 * np.mean(np.abs(resid) / np.maximum(np.abs(yt), EPS)),
        })
    return pd.DataFrame(rows)

def find_latest_run_log(sim, field, outkey):
    """
    runs/case_*_{sim}_{field}_{outkey}/{timestamp}/training_log.csv 를 탐색하여
    가장 최신 log 경로를 반환. 없으면 None.
    """
    parent_glob = os.path.join(RUNS_DIR, f"case_*_{sim}_{field}_{outkey}")
    parents = [p for p in glob.glob(parent_glob) if os.path.isdir(p)]
    candidates = []
    for p in parents:
        for sub in glob.glob(os.path.join(p, "*")):
            logp = os.path.join(sub, "training_log.csv")
            if os.path.isfile(logp):
                candidates.append(logp)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def find_latest_pred_subdir(sim, field, outkey):
    """
    predictions에서 해당 조합의 최신 타임스탬프 폴더를 반환.
    - 지원 형식 (부모 폴더):
        predictions/case_*_{sim}_{field}_{outkey}/<ts>/
        predictions/Case_*_{sim}_{field}_{outkey}/<ts>/
    """
    bases = []
    bases += glob.glob(os.path.join(PREDS_DIR, f"case_*_{sim}_{field}_{outkey}"))
    bases += glob.glob(os.path.join(PREDS_DIR, f"Case_*_{sim}_{field}_{outkey}"))
    subdirs = []
    for b in bases:
        if not os.path.isdir(b):
            continue
        for sd in glob.glob(os.path.join(b, "*")):
            if os.path.isdir(sd) and os.path.isfile(os.path.join(sd, "preds.npy")) and os.path.isfile(os.path.join(sd, "trues.npy")):
                subdirs.append(sd)
    if not subdirs:
        return None
    # 타임스탬프 폴더의 mtime 기준 최신
    return max(subdirs, key=os.path.getmtime)

def param_names_by_key(outkey, out_dim):
    if out_dim == 2:
        if outkey == "A":
            return ["Omega_m", "sigma_8"]
        if outkey == "B":
            return ["AGN1", "AGN2"]
        if outkey == "C":
            return ["SN1", "SN2"]
        return ["param1", "param2"]
    elif out_dim == 6:
        return ["Omega_m", "sigma_8", "SN1", "AGN1", "SN2", "AGN2"]
    else:
        return [f"param{i}" for i in range(out_dim)]

def main():
    start_t = time.perf_counter()

    # 12개 타깃 조합 (고정 순서)
    input_order = [("SIMBA", "Mtot"), ("SIMBA", "P"),
                   ("TNG", "Mtot"), ("TNG", "P")]
    out_order = ["A", "B", "C"]

    # predictions에서 각 조합의 최신 폴더 수집
    cases = []  # list of ((sim, field, outk), pred_ts_dir)
    missing = []
    for sim, field in input_order:
        for outk in out_order:
            ts_dir = find_latest_pred_subdir(sim, field, outk)
            if ts_dir is None:
                missing.append((sim, field, outk))
            else:
                cases.append(((sim, field, outk), ts_dir))

    print(f"발견된 케이스: {len(cases)} / 12")
    if missing:
        print("누락 케이스:", ", ".join([f"{s}_{f}_{o}" for s, f, o in missing]))

    # =============================
    # 1. Loss Curves (3x4, 누락도 자리 유지)
    # =============================
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, (sim, field, outk) in enumerate([(s, f, o) for s, f in input_order for o in out_order]):
        ax = axes[idx]
        if ((sim, field, outk),) not in [((k[0], k[1], k[2]),) for k, _ in cases]:
            # 자리 채우기
            ax.text(0.5, 0.5, f"MISSING:\n{sim}_{field}_{outk}", ha="center", va="center", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{sim}_{field}_{outk}", fontsize=9)
            continue

        # 해당 케이스의 run log 찾기
        log_path = find_latest_run_log(sim, field, outk)
        if log_path is None:
            ax.text(0.5, 0.5, f"No training_log.csv\n{sim}_{field}_{outk}", ha="center", va="center", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{sim}_{field}_{outk}", fontsize=9)
            continue

        log_df = load_training_log(log_path)
        if log_df is None:
            ax.text(0.5, 0.5, f"Load error\n{os.path.basename(log_path)}", ha="center", va="center", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{sim}_{field}_{outk}", fontsize=9)
            continue

        if {"epoch", "train_loss", "val_loss"}.issubset(log_df.columns):
            ax.plot(log_df["epoch"], log_df["train_loss"], label="Train", color="blue")
            ax.plot(log_df["epoch"], log_df["val_loss"], label="Val", color="orange")
        elif {"epoch", "loss"}.issubset(log_df.columns):
            ax.plot(log_df["epoch"], log_df["loss"], label="Loss", color="blue")
        else:
            ax.text(0.5, 0.5, "Columns not found", ha="center", va="center", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        ax.set_title(f"{sim}_{field}_{outk}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=6)

    plt.tight_layout()
    loss_png = os.path.join(SAVE_DIR, f"loss_curves_all_cases_{RUN_TS}.png")
    plt.savefig(loss_png, dpi=200)
    plt.close()

    # =============================
    # 2. Scatter Plots (3x4, 누락도 자리 유지)
    # =============================
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    summary_rows = []

    # 고정 순서로 루프 돌며 자리 맞추기
    lookup = { (sim, field, outk): ts_dir for (sim, field, outk), ts_dir in cases }
    for idx, (sim, field, outk) in enumerate([(s, f, o) for s, f in input_order for o in out_order]):
        ax = axes[idx]
        ts_dir = lookup.get((sim, field, outk))
        if ts_dir is None:
            ax.text(0.5, 0.5, f"MISSING:\n{sim}_{field}_{outk}", ha="center", va="center", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{sim}_{field}_{outk}", fontsize=9)
            continue

        preds = load_predictions(ts_dir)
        if preds is None:
            ax.text(0.5, 0.5, f"No preds/trues\n{sim}_{field}_{outk}", ha="center", va="center", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{sim}_{field}_{outk}", fontsize=9)
            continue

        y_true, y_pred = preds
        out_dim = y_true.shape[1]
        param_names = param_names_by_key(outk, out_dim)

        # metrics 저장
        metrics_df = compute_metrics(y_true, y_pred, param_names)
        # 열을 평탄화해서 한 행으로 저장: param별 dict -> 접두사 붙이기
        flat = {"Case": f"{sim}_{field}_{outk}"}
        for _, row in metrics_df.iterrows():
            p = row["param"]
            for col in ["R2", "RMSE", "MAE", "Pearson_r", "Spearman_rho", "Bias", "Std", "RelErr"]:
                flat[f"{p}.{col}"] = row[col]
        summary_rows.append(flat)

        # 첫 번째 파라미터 산점도
        yt, yp = y_true[:, 0], y_pred[:, 0]
        sns.scatterplot(x=yt, y=yp, ax=ax, s=10, alpha=0.5)
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_title(f"{sim}_{field}_{outk}", fontsize=9)
        ax.set_xlabel(f"True {param_names[0]}")
        ax.set_ylabel(f"Pred {param_names[0]}")

    plt.tight_layout()
    scatter_png = os.path.join(SAVE_DIR, f"scatter_all_cases_{RUN_TS}.png")
    plt.savefig(scatter_png, dpi=200)
    plt.close()

    # =============================
    # 3. Metrics Summary 저장 (+ 실행 시간)
    # =============================
    summary_df = pd.DataFrame(summary_rows)

    end_t = time.perf_counter()
    runtime_sec = end_t - start_t
    ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_df["EvalRuntimeSec"] = runtime_sec
    summary_df["EvalEndedAt"] = ended_at

    out_csv = os.path.join(SAVE_DIR, f"metrics_summary_all_cases_{RUN_TS}.csv")
    summary_df.to_csv(out_csv, index=False)

    print("✅ 완료! 결과는:")
    print(f"- Loss curves: {loss_png}")
    print(f"- Scatter plots: {scatter_png}")
    print(f"- Metrics table: {out_csv}")
    print(f"- Eval runtime: {runtime_sec:.3f} sec (ended at {ended_at})")

if __name__ == "__main__":
    main()
