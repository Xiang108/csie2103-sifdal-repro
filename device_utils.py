#!/usr/bin/env python3
"""依 GPU 占用情況選擇 cuda 或 cpu。"""
from __future__ import annotations

import subprocess


def _gpu_stats() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    stats = []
    for line in out.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        stats.append({"index": int(idx), "mem_mb": int(mem), "util": int(util)})
    return stats


def choose_device(
    prefer_gpu: bool = True,
    max_util: int = 85,
    max_mem_mb: int = 8000,
    gpu_id: int | None = None,
) -> tuple[bool, str]:
    """回傳 (use_cuda, reason)。"""
    stats = _gpu_stats()
    if not stats:
        return False, "未偵測到 GPU，使用 CPU"

    if gpu_id is not None:
        g = next((s for s in stats if s["index"] == gpu_id), None)
        if g is None:
            return False, f"指定 GPU{gpu_id} 不存在，使用 CPU"
        if prefer_gpu:
            return True, f"使用指定 GPU{gpu_id}（mem={g['mem_mb']}MiB util={g['util']}%）"
        return False, f"prefer_gpu=False，使用 CPU"

    for g in stats:
        if g["util"] <= max_util and g["mem_mb"] <= max_mem_mb:
            return True, (
                f"使用 GPU{g['index']}（mem={g['mem_mb']}MiB util={g['util']}%）"
            )

    # 全部忙碌時仍用占用較低者
    best = min(stats, key=lambda s: (s["util"], s["mem_mb"]))
    if prefer_gpu:
        return True, (
            f"GPU 皆忙碌，選 GPU{best['index']}（mem={best['mem_mb']}MiB util={best['util']}%）"
        )
    return False, "prefer_gpu=False，使用 CPU"
