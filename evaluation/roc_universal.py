import json
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm
import pickle
import os
import argparse

# Compute IoU between two xyxy boxes
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / (union + 1e-6)

# Load GT COCO
def load_ground_truth(coco_json_path):
    with open(coco_json_path, "r") as f:
        data = json.load(f)

    # images = list of dicts
    # id values are strings in your dataset → ok
    images = data["images"]

    # map image_id --> list of GT boxes
    gt_dict = {img["id"]: [] for img in images}

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        img_id = ann["image_id"]

        # convert to xyxy
        box = [x, y, x + w, y + h]
        gt_dict[img_id].append(box)

    return gt_dict, images

# Load predictions (COCO detection format)
def load_predictions(pred_json_path):
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    pred_dict = {}

    for p in preds:
        img_id = p["image_id"]
        x, y, w, h = p["bbox"]
        score = p["score"]

        box = [x, y, x + w, y + h]  # convert to xyxy

        if img_id not in pred_dict:
            pred_dict[img_id] = []
        pred_dict[img_id].append([score, box])

    return pred_dict

# Build frame-level labels and scores
def build_frame_scores(gt_dict, pred_dict, img_list, iou_thr=0.2):
    true_labels = []
    pred_scores = []

    for img in tqdm(img_list, desc="Processing frames"):
        img_id = img["id"]
        gt_boxes = gt_dict[img_id]
        preds = pred_dict.get(img_id, [])

        # Does this frame contain a polyp?
        frame_has_polyp = 1 if len(gt_boxes) > 0 else 0
        true_labels.append(frame_has_polyp)

        if len(preds) == 0:
            pred_scores.append(0)
            continue

        best_score = 0

        if frame_has_polyp == 0:
            # No GT: take the max confidence
            best_score = max([p[0] for p in preds])
        else:
            # GT exists: look for detections with IoU ≥ threshold
            for score, box_pred in preds:
                for box_gt in gt_boxes:
                    if compute_iou(box_pred, box_gt) >= iou_thr:
                        best_score = max(best_score, score)
                        break

        pred_scores.append(best_score)

    return np.array(true_labels), np.array(pred_scores)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("coco_gt", type=str, help="COCO GT json file")
    parser.add_argument("pred_json", type=str, help="Predictions json")
    parser.add_argument("output_dir", type=str, help="Directory to save ROC + PKL")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ground truth...")
    gt_dict, img_list = load_ground_truth(args.coco_gt)

    print("Loading predictions...")
    pred_dict = load_predictions(args.pred_json)

    print("Building frame-level labels and scores...")
    y_true, y_score = build_frame_scores(gt_dict, pred_dict, img_list, iou_thr=0.2)

    # save pkl
    pkl_path = os.path.join(args.output_dir, "labels_scores.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump((y_true, y_score), f)
    print("Saved:", pkl_path)

    # compute ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)
    print("AUC =", auc_value)

    # save ROC figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Frame-level ROC (IoU ≥ 0.2)")
    plt.grid(True)
    plt.legend()
    fig_path = os.path.join(args.output_dir, "ROC_curve.png")
    plt.savefig(fig_path, dpi=300)
    print("ROC saved to:", fig_path)
