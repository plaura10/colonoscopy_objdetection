#!/usr/bin/env python3
"""
Sampling script for REAL-Colon COCO-style annotations.

TRAIN:
    - Keep ALL positive frames.
    - Sample a fixed number of negative frames per video (neg_per_video_train).

VALIDATION:
    - Sample a fixed number of negative frames per video (neg_per_video_val).
    - Sample a fixed number of positive frames per video (pos_per_video_val),
      only from videos containing positive frames.
    - Example goal: 12,000 frames total (7,440 negatives, 4,560 positives).

TEST:
    - Left unchanged (full dataset).
"""

import json
import os
import random
from collections import Counter


# CONFIGURATION (edit these paths and parameters)
TRAIN_JSON_PATH = "train_ann.json"
VAL_JSON_PATH = "val_ann.json"
TEST_JSON_PATH = "test_ann.json"

OUT_TRAIN_JSON = "train_sampled.json"
OUT_VAL_JSON = "val_sampled.json"
OUT_TEST_JSON = "test_full.json"

NEG_PER_VIDEO_TRAIN = 3117     # negative frames per video in TRAIN
NEG_PER_VIDEO_VAL = 620        # negative frames per video in VALIDATION
POS_PER_VIDEO_VAL = 456        # positive frames per video in VALIDATION

RANDOM_SEED = 42

def load_coco(path):
    """Load a COCO-style JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_coco(data, path):
    """Save a COCO-style dictionary as JSON."""
    with open(path, "w") as f:
        json.dump(data, f)


def extract_video_id(file_name):
    """
    Extract video ID from filename.
    Assumes format: '0YZ-0BC_000123.jpg' → video ID = '0YZ-0BC'
    """
    return file_name.split("_")[0]

def summarize_split(name, coco_json):
    """Print basic statistics for a COCO split."""
    images = coco_json.get("images", [])
    annotations = coco_json.get("annotations", [])

    n_images = len(images)
    n_annotations = len(annotations)

    annotated_image_ids = {ann["image_id"] for ann in annotations}
    n_pos_images = len(annotated_image_ids)
    n_neg_images = n_images - n_pos_images

    print(f"\n=== {name} ===")
    print(f"Images:       {n_images}")
    print(f"Annotations:  {n_annotations}")
    print(f"Pos. images:  {n_pos_images}")
    print(f"Neg. images:  {n_neg_images}")

    return {
        "images": n_images,
        "annotations": n_annotations,
        "pos_images": n_pos_images,
        "neg_images": n_neg_images,
    }


def sanity_checks(name, coco_json):
    """Check for missing references, duplicates, and annotation consistency."""
    images = coco_json["images"]
    annotations = coco_json["annotations"]

    image_ids = {im["id"] for im in images}
    ann_image_ids = {ann["image_id"] for ann in annotations}

    missing = ann_image_ids - image_ids
    if missing:
        print(f"[{name}] ERROR: annotations reference missing images: {missing}")
    else:
        print(f"[{name}] OK: all annotations reference valid images.")

    counts = Counter(im["id"] for im in images)
    dup = [img_id for img_id, c in counts.items() if c > 1]
    if dup:
        print(f"[{name}] WARNING: duplicated image IDs detected: {dup}")
    else:
        print(f"[{name}] OK: no duplicated image IDs.")

# Sampling Functions
def filter_pos_frames(coco_json):
    """
    Return only frames containing at least one annotation (positive frames)
    and their corresponding annotations.
    """
    annotated_image_ids = {ann["image_id"] for ann in coco_json["annotations"]}

    filtered_images = [im for im in coco_json["images"] if im["id"] in annotated_image_ids]
    filtered_ann = [ann for ann in coco_json["annotations"] if ann["image_id"] in annotated_image_ids]

    return {
        "info": coco_json.get("info", {}),
        "licenses": coco_json.get("licenses", []),
        "categories": coco_json.get("categories", []),
        "images": filtered_images,
        "annotations": filtered_ann,
    }


def filter_neg_frames(coco_json, neg_per_video=3117, seed=42):
    """
    TRAIN SAMPLING:
        - Keep all positive frames.
        - Sample up to neg_per_video negative frames per video.
    """
    random.seed(seed)

    annotated_image_ids = {ann["image_id"] for ann in coco_json["annotations"]}

    pos_images = [im for im in coco_json["images"] if im["id"] in annotated_image_ids]
    neg_images = [im for im in coco_json["images"] if im["id"] not in annotated_image_ids]

    pos_ann = [ann for ann in coco_json["annotations"] if ann["image_id"] in annotated_image_ids]

    # Group negatives per video
    groups = {}
    for im in neg_images:
        video_id = extract_video_id(im["file_name"])
        groups.setdefault(video_id, []).append(im)

    neg_final = []
    for video_id, imgs in groups.items():
        if len(imgs) >= neg_per_video:
            sampled = random.sample(imgs, neg_per_video)
        else:
            sampled = imgs.copy()
            print(f"[TRAIN] Warning: video {video_id} has only {len(imgs)} negatives (required {neg_per_video})")
        neg_final.extend(sampled)

    new_images = pos_images + neg_final
    random.shuffle(new_images)

    return {
        "info": coco_json.get("info", {}),
        "licenses": coco_json.get("licenses", []),
        "categories": coco_json.get("categories", []),
        "images": new_images,
        "annotations": pos_ann,
    }


def filter_frames_val(coco_json, neg_per_video=620, pos_per_video=456, seed=42):
    """
    VALIDATION SAMPLING:
        - Sample fixed number of negative frames per video.
        - Sample fixed number of positive frames per video, only for videos containing positives.
        - Keep only annotations of selected positive frames.
    """
    random.seed(seed)

    annotated_image_ids = {ann["image_id"] for ann in coco_json["annotations"]}

    pos_images = [im for im in coco_json["images"] if im["id"] in annotated_image_ids]
    neg_images = [im for im in coco_json["images"] if im["id"] not in annotated_image_ids]

    # Group per video
    groups_neg = {}
    for im in neg_images:
        video_id = extract_video_id(im["file_name"])
        groups_neg.setdefault(video_id, []).append(im)

    groups_pos = {}
    for im in pos_images:
        video_id = extract_video_id(im["file_name"])
        groups_pos.setdefault(video_id, []).append(im)

    # Sample negatives
    neg_final = []
    for video_id, imgs in groups_neg.items():
        if len(imgs) >= neg_per_video:
            sampled = random.sample(imgs, neg_per_video)
        else:
            sampled = imgs.copy()
            print(f"[VAL] Warning: video {video_id} has only {len(imgs)} negatives (required {neg_per_video})")
        neg_final.extend(sampled)

    # Sample positives
    pos_final = []
    for video_id, imgs in groups_pos.items():
        if len(imgs) == 0:
            continue  # no positives in this video
        if len(imgs) >= pos_per_video:
            sampled = random.sample(imgs, pos_per_video)
        else:
            sampled = imgs.copy()
            print(f"[VAL] Warning: video {video_id} has only {len(imgs)} positives (required {pos_per_video})")
        pos_final.extend(sampled)

    # Keep only annotations for selected positive images
    selected_pos_ids = {im["id"] for im in pos_final}
    pos_ann = [ann for ann in coco_json["annotations"] if ann["image_id"] in selected_pos_ids]

    new_images = pos_final + neg_final
    random.shuffle(new_images)

    return {
        "info": coco_json.get("info", {}),
        "licenses": coco_json.get("licenses", []),
        "categories": coco_json.get("categories", []),
        "images": new_images,
        "annotations": pos_ann,
    }

def main():
    # Load original splits
    train = load_coco(TRAIN_JSON_PATH)
    val = load_coco(VAL_JSON_PATH)
    test = load_coco(TEST_JSON_PATH)

    # Show summary of original sets
    summarize_split("TRAIN (original)", train)
    summarize_split("VAL (original)", val)
    summarize_split("TEST (original)", test)

    # TRAIN SAMPLING
    train_sampled = filter_neg_frames(
        train,
        neg_per_video=NEG_PER_VIDEO_TRAIN,
        seed=RANDOM_SEED
    )
    summarize_split("TRAIN (sampled)", train_sampled)
    sanity_checks("TRAIN (sampled)", train_sampled)
    save_coco(train_sampled, OUT_TRAIN_JSON)

    # VALIDATION SAMPLING
    val_sampled = filter_frames_val(
        val,
        neg_per_video=NEG_PER_VIDEO_VAL,
        pos_per_video=POS_PER_VIDEO_VAL,
        seed=RANDOM_SEED
    )
    summarize_split("VAL (sampled)", val_sampled)
    sanity_checks("VAL (sampled)", val_sampled)
    save_coco(val_sampled, OUT_VAL_JSON)

    # TEST SET: unchanged
    save_coco(test, OUT_TEST_JSON)
    summarize_split("TEST (saved full)", test)

    print("\nSampling completed.")
    print("Output files:")
    print(f"  {OUT_TRAIN_JSON}")
    print(f"  {OUT_VAL_JSON}")
    print(f"  {OUT_TEST_JSON}")


if __name__ == "__main__":
    main()
