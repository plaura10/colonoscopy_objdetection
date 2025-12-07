# YOLOv7 – Training Configuration for REAL-Colon

This folder contains the **modified YOLOv7 script and configuration files** that were used to train YOLOv7 on the REAL-Colon dataset for the polyp detection experiments in this project.

Only the files that differ from the original repository are included here.

## 1. Base Repository
The implementation is based on the official YOLOv7 repository: https://github.com/WongKinYiu/yolov7.
For colonoscopy-specific extensions, we referenced the fork: https://github.com/cosmoimd/yolov7.

## 2. Modifications Introduced
### Removal of the `negative_sampling_rate` parameter
The `negative_sampling_rate` argument was **removed** from `train.py` because it is defined in the original repository but **never implemented** in the sampling or dataloader logic, leaving it enabled leads to errors during training. This change does not alter the YOLOv7 training logic.

### Adaptation of `test.py` for colonoscopy evaluation
The colonoscopy YOLOv7 fork already includes logic to automatically detect whether the dataset refers to *polyp* or *lesion* detection and import the correct colonoscopy-specific COCOeval implementation accordingly. Therefore, no change to test.py was needed.

### Custom dataset configuration
A dataset YAML file (`real_colon_dataset.yaml`) is included, containing:

- paths to the YOLO-ready REAL-Colon dataset (after sampling),
- number of classes,
- class definitions.

## 3. How to Use These Files
### Step 1 — Clone YOLOv7
```bash
git clone https://github.com/cosmoimd/yolov7
cd yolov7
```

### Step 2 - Apply the modified files 
Replace the original files in YOLOv7 with the modified versions:
```bash
cp ../models/yolov7/train.py ./train.py
cp ../models/yolov7/real_colon_dataset.yaml ./data/
```

### Step 3 - Training YOLOv7 on REAL-Colon
```bash
python train.py \
    --workers 8 \
    --device 0 \
    --data  <path_to_yaml_file> \
    --img 640 \
    --batch 16 \
    --epochs 50 \
    --hyp data/hyp.scratch.colonoscopy.yaml \
    --cfg cfg/training/yolov7_colon.yaml \
    --name <model_name> \
    --weights yolov7.pt
```

This will save model weights in: ```runs/train/<experiment_name>/```.

### Step 4 - Fine-Tuning 
In addition to the code modifications, this project included an extensive fine-tuning phase, during which different training hyperparameters were explored.
The goal was to identify the configuration that achieved the highest mAP<sub>0.5:0.95</sub> on the REAL-Colon validation set and to evaluate only this best-performing configuration on the test set to compare performance with the other architectures.

Several versions of the file hyp.scratch.colonoscopy.yaml were used to test different hyperparameter values.
Across multiple runs, various parameters were tuned, including: image size, batch size, learning rate, the presence of data augmentation, initialization, scheduler, optimizer, percentage of negative frames (by creating another dataset with only positive frame on the training set). 

The individual experiment setups are not included in this repository, as they represent experiment-specific configurations rather than modifications to the YOLOv7 codebase.
However, the fine-tuning process can be reproduced by adjusting the hyperparameter values in:
```data/hyp.scratch.colonoscopy.yaml``` and running the YOLOv7 training commands described in this README.
