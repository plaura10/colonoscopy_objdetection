#!/usr/bin/env python3
"""
Build YOLO-ready dataset after sampling.

This script performs the final step in the REAL-Colon formatting pipeline:

1. Reads the sampled COCO annotation files produced by sampling_real_colon.py
2. Copies ONLY the images referenced in train_sampled.json and val_sampled.json
3. Copies the corresponding YOLO label (.txt) files
4. Creates a clean YOLO dataset structure:
       final_yolo/
           train/images/
           train/labels/
           train/train_ann.json
           val/images/
           val/labels/
           val/val_ann.json
5. Runs consistency checks to ensure:
       - all images listed in the JSON exist
       - each image has the corresponding .txt label
       - no label is missing its image
"""

import os
import json
import shutil
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/path/to/your/dir")

# Folder containing the sampled JSON output
SAMPLED_FOLDER = BASE_DIR / "dataset"

# Folder produced by export_yolo_coco_format.py (contains ALL images/labels)
ORIGINAL_SPLIT_FOLDER = BASE_DIR / "split"

# Output folder for the final YOLO dataset
FINAL_YOLO_FOLDER = BASE_DIR / "final_yolo"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# STEP 0 — COPY train_ann.json / val_ann.json
def copy_json(split):
    print(f"\n Copying JSON for {split}...")

    src = SAMPLED_FOLDER / split / f"{split}_ann.json"
    dst = FINAL_YOLO_FOLDER / split / f"{split}_ann.json"

    ensure_dir(dst.parent)

    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied {src.name} → {dst}")
    else:
        print(f"JSON not found: {src}")

# STEP 1: COPY ONLY FILTERED IMAGES
def copy_filtered_images(split):
    print(f"\n Copying images for {split}...")

    json_path = SAMPLED_FOLDER / split / f"{split}_ann.json"
    original_img_dir = ORIGINAL_SPLIT_FOLDER / split / "images"
    target_dir = FINAL_YOLO_FOLDER / split / "images"

    ensure_dir(target_dir)

    data = load_json(json_path)
    filenames = {img["file_name"] for img in data["images"]}

    copied = 0
    missing = []

    for fname in filenames:
        src = original_img_dir / fname
        dst = target_dir / fname

        if src.exists():
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append(fname)

    print(f"Copied {copied} images.")
    if missing:
        print(f"Missing {len(missing)} images:", missing[:10])

# STEP 2: COPY ONLY CORRESPONDING YOLO LABEL FILES
def copy_filtered_labels(split):
    print(f"\n Copying labels for {split}...")

    json_path = SAMPLED_FOLDER / split / f"{split}_ann.json"
    original_label_dir = ORIGINAL_SPLIT_FOLDER / split / "labels"
    target_dir = FINAL_YOLO_FOLDER / split / "labels"

    ensure_dir(target_dir)

    data = load_json(json_path)
    label_names = {img["file_name"].replace(".jpg", ".txt") for img in data["images"]}

    copied = 0
    missing = []

    for fname in label_names:
        src = original_label_dir / fname
        dst = target_dir / fname

        if src.exists():
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append(fname)

    print(f"Copied {copied} labels.")
    if missing:
        print(f"Missing {len(missing)} labels:", missing[:10])


# STEP 3: CONSISTENCY CHECK
def check_consistency(split):
    print(f"\n Checking consistency for {split}...")

    json_path = SAMPLED_FOLDER / split / f"{split}_ann.json"
    img_dir = FINAL_YOLO_FOLDER / split / "images"
    lbl_dir = FINAL_YOLO_FOLDER / split / "labels"

    if not json_path.exists():
        print(f"Missing JSON: {json_path}")
        return

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Missing directories for split: {split}")
        return

    data = load_json(json_path)

    json_imgs = len(data["images"])
    fs_imgs = len(list(img_dir.glob("*.jpg")))
    fs_lbls = len(list(lbl_dir.glob("*.txt")))

    print(f"Images in JSON:   {json_imgs}")
    print(f"Images copied:    {fs_imgs}")
    print(f"Labels copied:    {fs_lbls}")

    if json_imgs != fs_imgs:
        print("Image count mismatch.")
    if json_imgs != fs_lbls:
        print("Label count mismatch.")

    # Check images without labels
    missing_labels = [
        img.name for img in img_dir.glob("*.jpg")
        if not (lbl_dir / f"{img.stem}.txt").exists()
    ]

    if missing_labels:
        print(f"{len(missing_labels)} images without labels.")
        print("Examples:", missing_labels[:10])
    else:
        print("All images have labels.")

    # Check labels without images
    missing_images = [
        lbl.name for lbl in lbl_dir.glob("*.txt")
        if not (img_dir / f"{lbl.stem}.jpg").exists()
    ]

    if missing_images:
        print(f"{len(missing_images)} labels without images.")
        print("Examples:", missing_images[:10])
    else:
        print("All labels have images.")


def main():
    print("\nBuilding YOLO-ready dataset...")

    for split in ["train", "val"]:
        copy_json(split)
        copy_filtered_images(split)
        copy_filtered_labels(split)

    print("\nRunning consistency checks...")
    for split in ["train", "val"]:
        check_consistency(split)

    print("\nYOLO dataset successfully generated")
    print(f"Location: {FINAL_YOLO_FOLDER}")

if __name__ == "__main__":
    main()
