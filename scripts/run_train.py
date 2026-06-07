#!/usr/bin/env python3
# 建議使用 /usr/bin/python3（含 numpy/torch）；miniconda base 可能缺套件。
"""SIFDAL 完整 VisDrone 訓練（預設 300 epoch，對齊論文）+ 結果寫入 results.json。"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIFAD = ROOT / "sifdal" / "SIFAD"
REPRO = ROOT
RESULT = REPRO / "results.json"


def _log_path() -> Path:
    epochs = int(os.environ.get("SIFDAL_EPOCHS", "300"))
    return REPRO / "outputs" / f"train_{epochs}e.log"


def patch_train_py():
    p = SIFAD / "train.py"
    text = p.read_text(encoding="utf-8")
    total_epochs = int(os.environ.get("SIFDAL_EPOCHS", "300"))
    freeze_all = os.environ.get("SIFDAL_FREEZE_ALL", "1") == "1"
    freeze_epochs = int(os.environ.get("SIFDAL_FREEZE_EPOCH", str(total_epochs if freeze_all else 50)))
    init_lr = os.environ.get("SIFDAL_INIT_LR", "1e-3")
    if freeze_all:
        freeze_epochs = total_epochs

    if "from device_utils import choose_device" not in text:
        text = text.replace(
            "import datetime\nimport os\nfrom functools import partial",
            "import datetime\nimport os\nimport sys\nfrom functools import partial\n\n"
            "_REPRO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"..\"))\n"
            "if _REPRO_ROOT not in sys.path:\n    sys.path.insert(0, _REPRO_ROOT)\n"
            "from device_utils import choose_device  # noqa: E402",
        )

    replacements = [
        ("Cuda            = True", "Cuda, _device_msg = choose_device(gpu_id=int(os.environ.get('SIFDAL_GPU', '1')))\n    print(f'[Device] {_device_msg}')"),
        ("model_path      = 'model_data/yolov7_weights.pth'", "model_path      = ''"),
        ("pretrained      = False", "pretrained      = True"),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)

    text = re.sub(r"Freeze_Epoch\s*=\s*\d+", f"Freeze_Epoch        = {freeze_epochs}", text)
    text = re.sub(r"UnFreeze_Epoch\s*=\s*\d+", f"UnFreeze_Epoch      = {total_epochs}", text)
    text = re.sub(r"Unfreeze_batch_size\s*=\s*\d+", "Unfreeze_batch_size = 8", text)
    text = re.sub(r"Freeze_batch_size\s*=\s*\d+", "Freeze_batch_size   = 8", text)
    text = re.sub(r"Init_lr\s*=\s*[\d.e-]+", f"Init_lr             = {init_lr}", text)
    text = re.sub(r"save_period\s*=\s*\d+", "save_period         = 10", text)
    if "choose_device" in text and "Freeze_Train        = True" not in text:
        pass
    text = re.sub(
        r"Freeze_Train\s*=\s*\w+.*",
        "Freeze_Train        = True  # 論文：先凍結再解凍；SIFDAL_FREEZE_ALL=1 則全程凍結",
        text,
        count=1,
    )

    # 單 GPU，不用 DataParallel
    text = text.replace(
        "            model_train = torch.nn.DataParallel(model)\n            cudnn.benchmark = True\n            model_train = model_train.cuda()",
        "            model_train = model_train.cuda()\n            cudnn.benchmark = True",
    )
    p.write_text(text, encoding="utf-8")


def patch_utils_fit():
    p = SIFAD / "utils" / "utils_fit.py"
    text = p.read_text(encoding="utf-8")
    if "_scalar_loss" not in text:
        text = text.replace(
            "from utils.utils import get_lr\n\n\n#---------------------------------------#",
            "from utils.utils import get_lr\n\n\ndef _scalar_loss(loss):\n    return loss.mean() if loss.dim() != 0 else loss\n\n\n#---------------------------------------#",
        )
        text = text.replace(
            "            loss_value = loss_value + loss_accr\n\n            # ----------------------#\n            #   反向传播",
            "            loss_value = _scalar_loss(loss_value + loss_accr)\n\n            # ----------------------#\n            #   反向传播",
        )
        text = text.replace(
            "                loss_value = loss_value + loss_accr\n\n            # ----------------------#\n            #   反向传播\n            # ----------------------#\n            scaler.scale(loss_value).backward()",
            "                loss_value = _scalar_loss(loss_value + loss_accr)\n\n            # ----------------------#\n            #   反向传播\n            # ----------------------#\n            scaler.scale(loss_value).backward()",
        )
    p.write_text(text, encoding="utf-8")


def patch_callbacks():
    p = SIFAD / "utils" / "callbacks.py"
    text = p.read_text(encoding="utf-8")
    if "def _f(x):" not in text:
        text = text.replace(
            "        self.losses.append(loss)\n        self.val_loss.append(val_loss)\n        self.accr_loss.append(accr_loss)",
            "        def _f(x):\n            return float(x.detach().cpu()) if hasattr(x, 'detach') else float(x)\n\n        loss, val_loss, accr_loss = _f(loss), _f(val_loss), _f(accr_loss)\n        self.losses.append(loss)\n        self.val_loss.append(val_loss)\n        self.accr_loss.append(accr_loss)",
        )
    p.write_text(text, encoding="utf-8")


def parse_results() -> dict:
    target = int(os.environ.get("SIFDAL_EPOCHS", "300"))
    logs = sorted((SIFAD / "logs").glob("loss_*"), key=lambda x: x.stat().st_mtime)
    if not logs:
        return {"status": "failed", "note": "無 logs 目錄"}
    log_dir = logs[-1]
    maps = []
    mp = log_dir / "epoch_map.txt"
    if mp.exists():
        maps = [float(x) for x in mp.read_text().strip().splitlines() if x.strip()]

    best = max(maps) if maps else 0.0
    final = maps[-1] if maps else 0.0
    log = _log_path()
    ran_epochs = 0
    if log.exists():
        hits = re.findall(rf"Epoch:(\d+)/{target}", log.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            ran_epochs = int(hits[-1])
    freeze_all = os.environ.get("SIFDAL_FREEZE_ALL", "1") == "1"
    collapsed = best > 0.2 and final < best * 0.75
    ok = ran_epochs >= target and final >= 0.15 and not collapsed
    return {
        "paper": "SIFDAL (IEEE TGRS 2025)",
        "dataset": "VisDrone2019-DET full train 6471 + val 548 (VOC)",
        "epochs": target,
        "epochs_completed": ok,
        "freeze_all": freeze_all,
        "collapse_detected": collapsed,
        "device": f"cuda:{os.environ.get('SIFDAL_GPU', '1')}",
        "log_dir": str(log_dir.relative_to(ROOT.parent)),
        "metrics": {
            "best_mAP": round(best, 4),
            "final_mAP": round(final, 4),
            "map_history": [round(m, 4) for m in maps[-20:]],
        },
        "figures_dir": "reproduction/csie2103-sifdal-repro/figures",
        "status": "completed" if ok else ("collapsed" if collapsed else "partial"),
        "note": None if ok else ("解凍後 mAP 崩潰" if collapsed else f"未跑滿 {target} epoch 或 mAP 偏低"),
        "log": str(_log_path().relative_to(ROOT.parent)),
    }


def main():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_visdrone.py")])
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "visdrone_to_voc.py")])
    REPRO.mkdir(parents=True, exist_ok=True)
    (REPRO / "outputs").mkdir(parents=True, exist_ok=True)

    n_train = len((SIFAD / "2007_train.txt").read_text().strip().splitlines())
    n_val = len((SIFAD / "2007_val.txt").read_text().strip().splitlines())
    if n_train < 6000:
        print(f"[WARN] train 僅 {n_train} 張，請先執行 visdrone_to_voc.py")

    patch_train_py()
    patch_utils_fit()
    patch_callbacks()

    env = os.environ.copy()
    gpu = env.get("SIFDAL_GPU", "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu

    epochs = int(os.environ.get("SIFDAL_EPOCHS", "300"))
    freeze_all = os.environ.get("SIFDAL_FREEZE_ALL", "1") == "1"
    freeze = int(os.environ.get("SIFDAL_FREEZE_EPOCH", str(epochs if freeze_all else 50)))
    log = _log_path()
    print(f"[SIFDAL] 開始 {epochs} epoch（freeze_all={freeze_all}, freeze={freeze}）GPU={gpu} train={n_train} val={n_val}")
    py = os.environ.get("SIFDAL_PYTHON", "/usr/bin/python3")
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            [py, "train.py"],
            cwd=SIFAD,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    log_tail = log.read_text(encoding="utf-8", errors="ignore")[-3000:]
    if proc.returncode != 0 or "Traceback" in log_tail:
        result = {"status": "failed", "log": str(log), "tail": log_tail[-800:]}
        RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(proc.returncode or 1)

    result = parse_results()
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "plot_figures.py")])
    subprocess.check_call([sys.executable, str(# fill_excel skipped)])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "completed":
        sys.exit(2)


if __name__ == "__main__":
    main()
