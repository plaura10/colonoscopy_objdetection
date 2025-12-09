# Polyp Detection 
This repository contains a complete and reproducible pipeline for benchmarking three object detection architectures on the **REAL-Colon dataset**:
- YOLOv7
- YOLOv11
- RT-DETR
  
The goal is to provide a standardized and comparable workflow for evaluating modern detectors in realistic colonoscopy settings.
This repository includes: 
1. Dataset preparation
2. Fine-tuning configuration for YOLOv7, YOLOv11 and RT-DETR
3. Experimental logs, plots and model output
4. Instruction to fully reproduce the experiments.

## 1. Dataset Preparation 
This module provides all scripts needed to prepare the REAL-Colon dataset for training YOLOv7, YOLOv11, and RT-DETR. It includes:
- Download and organize dataset files (frames, annotations, CSV metadata)
- Convert annotations to YOLO and COCO formats
- Sample positive and negative frames to create a filtered dataset
- Build final YOLO-ready and RT-DETR-ready datasets
For detailed instructions, scripts, and configuration options, see the [Dataset Preparation folder](https://github.com/plaura10/colonoscopy_objdetection/tree/main/dataset) 
