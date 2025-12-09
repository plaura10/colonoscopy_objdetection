# Polyp Detection 🩺
This repository contains a complete and reproducible pipeline for benchmarking three object detection architectures on the **REAL-Colon dataset**:

- YOLOv7  
- YOLOv11  
- RT-DETR  

The goal is to provide a standardized and comparable workflow for evaluating modern detectors in realistic colonoscopy settings.  
This repository includes:  

1. Dataset Preparation  
2. Model Training Configurations  
3. Model Evaluation  

---

## 1. Dataset Preparation 📁
This module provides all scripts needed to prepare the REAL-Colon dataset for training YOLOv7, YOLOv11, and RT-DETR. It includes:

- Download and organize dataset files (frames, annotations, CSV metadata)  
- Convert annotations to YOLO and COCO formats  
- Sample positive and negative frames to create a filtered dataset  
- Build final YOLO-ready and RT-DETR-ready datasets  

For detailed instructions, scripts, and configuration options, see the [Dataset Preparation folder](https://github.com/plaura10/colonoscopy_objdetection/tree/main/dataset).

---

## 2. Model Training Configurations 🤖
This module provides the necessary files and instructions to train the three object detection architectures on the REAL-Colon dataset.  

**Included models:**

- **YOLOv7:** Modified training scripts (`train.py`) and dataset YAML for colonoscopy-specific training.  
- **YOLOv11:** Dataset configuration and hyperparameters via Ultralytics CLI; no source code modifications required.  
- **RT-DETR:** Configuration files for training on the REAL-Colon dataset.  

For detailed instructions, scripts, and configuration options for each model, see the [Models folder](https://github.com/plaura10/colonoscopy_objdetection/tree/main/models).

---

## 3. Model Evaluation 📊
This module includes scripts and notebooks for:

- Evaluating model performance on the test set  
- Generating experimental logs and plots  
- Comparing YOLOv7, YOLOv11, and RT-DETR in realistic colonoscopy settings  

Evaluation results are reproducible following the full pipeline outlined [here](https://github.com/plaura10/colonoscopy_objdetection/tree/main/evaluation)
