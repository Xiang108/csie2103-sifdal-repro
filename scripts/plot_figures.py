#!/usr/bin/env python3
"""SIFDAL 重現：從 logs 產生報告用圖表。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT
SIFAD = ROOT / "sifdal" / "SIFAD"
OUT = REPRO / "figures"
RESULT = REPRO / "results.json"

PAPER_MAP = 0.40


def latest_log_dir() -> Path | None:
    logs = sorted(SIFAD.glob("logs/loss_*"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    metrics = result.get("metrics", {})
    log_dir = latest_log_dir()
    maps = []
    losses = []
    if log_dir:
        mp = log_dir / "epoch_map.txt"
        if mp.exists():
            maps = [float(x) for x in mp.read_text().splitlines() if x.strip()]
        lp = log_dir / "epoch_loss.txt"
        if lp.exists():
            losses = [float(x) for x in lp.read_text().splitlines() if x.strip()]

    best = metrics.get("best_mAP", max(maps) if maps else 0)
    final = metrics.get("final_mAP", maps[-1] if maps else 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if losses:
        axes[0].plot(range(1, len(losses) + 1), losses, color="#2563eb")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Train loss")
        axes[0].set_title("Training loss")
        axes[0].grid(alpha=0.3)
    if maps:
        axes[1].plot(range(len(maps)), [m * 100 for m in maps], "o-", color="#2563eb")
        axes[1].set_xlabel("Eval #")
        axes[1].set_ylabel("mAP (%)")
        axes[1].set_title("Validation mAP")
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "01_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Paper (~40%)", "Ours best", "Ours final"]
    vals = [PAPER_MAP * 100, float(best) * 100, float(final) * 100]
    colors = ["#16a34a", "#2563eb", "#94a3b8"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("mAP (%)")
    ax.set_title("VisDrone val mAP comparison")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "02_ap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    manifest = {"figures": ["01_training_curves.png", "02_ap_comparison.png"], "metrics": metrics}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
