#!/usr/bin/env python3
"""下載 VisDrone2019-DET（train + val）並解壓至 datasets/VisDrone-DET/。

來源：Ultralytics 鏡像（官方 VisDrone 發布之開放下載）
https://github.com/VisDrone/VisDrone-Dataset
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "datasets"))
DET = DATA_ROOT / "VisDrone-DET"
CACHE = DATA_ROOT / "_cache"

# 壓縮檔約 1.44 GB + 0.07 GB；解壓後 train+val 影像約 43 GB
URLS = {
    "VisDrone2019-DET-train.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        "VisDrone2019-DET-train.zip"
    ),
    "VisDrone2019-DET-val.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        "VisDrone2019-DET-val.zip"
    ),
}


def repo_root() -> Path:
    return REPO_ROOT


def is_ready() -> bool:
    train = DET / "VisDrone2019-DET-train" / "images"
    val = DET / "VisDrone2019-DET-val" / "images"
    return (
        train.is_dir()
        and val.is_dir()
        and len(list(train.glob("*.jpg"))) >= 6000
        and len(list(val.glob("*.jpg"))) >= 500
    )


def download_zip(name: str, url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip download] {name} ({dest.stat().st_size // 1_048_576} MB)")
        return dest
    print(f"[download] {name} ...")
    print(f"  URL: {url}")
    subprocess.check_call(["curl", "-L", "--retry", "3", "-o", str(dest), url])
    print(f"[done] {name} ({dest.stat().st_size // 1_048_576} MB)")
    return dest


def unzip_split(zip_path: Path, folder_name: str) -> None:
    target = DET / folder_name
    if (target / "images").is_dir() and list((target / "images").glob("*.jpg")):
        print(f"[skip unzip] {folder_name} 已存在")
        return
    DET.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path.name} -> {target}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DET)
    if not (target / "images").is_dir():
        raise SystemExit(f"解壓後找不到 {target}/images")


def ensure_det(force: bool = False) -> Path:
    if not force and is_ready():
        print(f"[OK] VisDrone-DET 已就緒：{DET}")
        return DET

    DET.mkdir(parents=True, exist_ok=True)
    train_zip = download_zip("VisDrone2019-DET-train.zip", URLS["VisDrone2019-DET-train.zip"])
    val_zip = download_zip("VisDrone2019-DET-val.zip", URLS["VisDrone2019-DET-val.zip"])
    unzip_split(train_zip, "VisDrone2019-DET-train")
    unzip_split(val_zip, "VisDrone2019-DET-val")

    n_train = len(list((DET / "VisDrone2019-DET-train" / "images").glob("*.jpg")))
    n_val = len(list((DET / "VisDrone2019-DET-val" / "images").glob("*.jpg")))
    print(f"[OK] VisDrone-DET train={n_train} val={n_val}")
    if n_train < 6000 or n_val < 500:
        raise SystemExit("資料集張數異常，請檢查下載是否完整")
    return DET


def ensure_coco(force: bool = False) -> Path:
    ensure_det(force=force)
    script = REPO_ROOT / "scripts" / "visdrone_to_coco.py"
    if not script.exists():
        raise SystemExit(f"找不到 {script}")
    cmd = [sys.executable, str(script)]
    if force:
        cmd.append("--force")
    subprocess.check_call(cmd)
    return DATA_ROOT / "VisDrone2019-DET-COCO"


def main():
    ap = argparse.ArgumentParser(description="下載 VisDrone2019-DET train+val")
    ap.add_argument("--coco", action="store_true", help="同時產生 COCO 格式（RemDet / Zoom-in）")
    ap.add_argument("--force", action="store_true", help="強制重新下載/轉換")
    args = ap.parse_args()

    ensure_det(force=args.force)
    if args.coco:
        ensure_coco(force=args.force)
    print("資料集準備完成。")


if __name__ == "__main__":
    main()
