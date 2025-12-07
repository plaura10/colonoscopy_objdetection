#!/usr/bin/env python3
"""
Convert COCO annotations (train/val/test) into RT-DETR–compatible format.

RT-DETR requires:
    - image IDs must be integers (0..N-1)
    - annotation IDs must be integers (0..M-1)
    - each annotation's image_id must match the remapped image IDs

This script:
1. Loads sampled COCO JSON (train/val/test)
2. Fixes image IDs and annotation IDs
3. Saves new COCO files in a dedicated folder:
        RTDETR/
            train.json
            val.json
            test.json
"""

import json
import os
from pathlib import Path

BASE_DIR = Path("/path/to/your/dir")

# Folder containing the sampled JSON output
SOURCE_FOLDER = BASE_DIR / "dataset"

# Output folder for RT-DETR compatible JSONs
OUTPUT_FOLDER = BASE_DIR / "RTDETR"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def fix_coco_ids(input_json_path, output_json_path):
    """Convert COCO JSON IDs so that image_id and annotation id are integer ranges."""
    with open(input_json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    id_map = {}                
    new_images = []

    for new_id, img in enumerate(images):
        old_id = img["id"]
        id_map[old_id] = new_id
        img["id"] = new_id
        new_images.append(img)

    data["images"] = new_images


    annotations = data.get("annotations", [])
    new_ann = []

    for new_ann_id, ann in enumerate(annotations):
        old_image_id = ann["image_id"]

        if old_image_id not in id_map:
            raise ValueError(
                f"Annotation refers to missing image_id {old_image_id}"
            )

        ann["id"] = new_ann_id            # new annotation ID
        ann["image_id"] = id_map[old_image_id]  # remapped image ID

        new_ann.append(ann)

    data["annotations"] = new_ann

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved RT-DETR formatted JSON → {output_json_path}")


def process_all_splits():
    """Apply ID normalization for train, val, test."""
    splits = ["train", "val", "test"]

    for split in splits:
        input_path = SOURCE_FOLDER / split / f"{split}_ann.json"
        output_path = OUTPUT_FOLDER / f"{split}_ann.json"

        if not input_path.exists():
            print(f"Skipping {split}: JSON not found → {input_path}")
            continue

        print(f"\nProcessing {split}...")
        fix_coco_ids(input_path, output_path)

    print("\n RT-DETR dataset successfully generated!")
    print(f"Location: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    process_all_splits()
