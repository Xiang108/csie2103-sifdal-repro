#!/usr/bin/env python3
"""VisDrone YOLO split → SIFDAL VOC 格式（含 2007_train.txt / 2007_val.txt）。"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIFAD = REPO / "sifdal" / "SIFAD"
VOC = SIFAD / "VOCdevkit" / "VOC2007"
DATA = Path(os.environ.get("DATA_ROOT", REPO / "datasets" / "VisDrone-DET"))

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]
# VisDrone class id → SIFDAL voc_classes.txt name
TO_VOC = {
    0: "pedestrian", 1: "pedestrian", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "bicycle", 7: "van", 8: "bus", 9: "motor",
}


def scale_class(w: int, h: int, img_w: int, img_h: int) -> int:
    area = (w * h) / max(img_w * img_h, 1)
    if area < 0.001:
        return 0
    if area < 0.01:
        return 1
    if area < 0.05:
        return 2
    if area < 0.15:
        return 3
    return 4


def yolo_to_voc(split: str):
    img_dir = DATA / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = DATA / f"VisDrone2019-DET-{split}" / "annotations"
    if not img_dir.exists():
        img_dir = DATA / "split" / split / "images"
        ann_dir = DATA / "split" / split / "labels"
    if not img_dir.exists():
        raise SystemExit(f"找不到 {split} 影像目錄，請先準備 {DATA}")

    (VOC / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (VOC / "Annotations").mkdir(parents=True, exist_ok=True)

    lines_out = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        lbl = ann_dir / f"{stem}.txt"
        if not lbl.exists():
            lbl = ann_dir / f"{img_path.name}.txt"
        if not lbl.exists():
            continue

        dst_img = VOC / "JPEGImages" / img_path.name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                import shutil
                shutil.copy2(img_path, dst_img)

        from PIL import Image
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = img_path.name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)
        ET.SubElement(size, "depth").text = "3"

        max_sc = 0
        bbox_parts: list[str] = []
        voc_classes = [
            x.strip()
            for x in (SIFAD / "model_data/voc_classes.txt").read_text().splitlines()
            if x.strip()
        ]
        for row in lbl.read_text(encoding="utf-8").splitlines():
            parts = row.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id not in TO_VOC:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            max_sc = max(max_sc, scale_class(w, h, img_w, img_h))
            name = TO_VOC[cls_id]
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = name
            ET.SubElement(obj, "difficult").text = "0"
            bb = ET.SubElement(obj, "bndbox")
            ET.SubElement(bb, "xmin").text = str(x1)
            ET.SubElement(bb, "ymin").text = str(y1)
            ET.SubElement(bb, "xmax").text = str(x2)
            ET.SubElement(bb, "ymax").text = str(y2)
            voc_id = voc_classes.index(name)
            bbox_parts.append(f"{x1},{y1},{x2},{y2},{voc_id}")

        if not bbox_parts:
            continue
        ET.SubElement(root, "scale_class").text = str(max_sc)
        xml_path = VOC / "Annotations" / f"{stem}.xml"
        ET.ElementTree(root).write(xml_path, encoding="utf-8")

        rel_img = f"VOCdevkit/VOC2007/JPEGImages/{img_path.name}"
        line = f"{rel_img} {max_sc} " + " ".join(bbox_parts)
        lines_out.append(line)

    out = SIFAD / f"2007_{split}.txt"
    out.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"{split}: {len(lines_out)} images -> {out}")


def main():
    if not DATA.exists():
        sys.exit(f"請先下載 VisDrone 至 {DATA}\n見 datasets/README.md")
    yolo_to_voc("train")
    yolo_to_voc("val")
    print("完成。執行: python3 scripts/run_train.py")


if __name__ == "__main__":
    main()
