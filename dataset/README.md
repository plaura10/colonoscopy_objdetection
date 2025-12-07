# Dataset Formatting 
This repository provides scripts to format the REAL-Colon dataset to the format for the models YOLOv7, YOLOv11 (YOLO format) and RT-DETR (COCO format). 

## 1. Dataset Downloads
First, download the REAL-Colon dataset at the link: https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866. 
After downloading, unzip all compressed files. 
In total, you will have 120 folders, since the dataset contains 60 videos and each video has two linked folders. One containing the frames in JPG format, the other one containing the annotations in XML format. 
The dataset also includes a lesion_info.csv file (information about the lesions) and video_info.csv file (information about the video).
 
## 2. Dataset Conversion and Split
This module allows you to convert the REAL-Colon dataset into **YOLO** and **COCO** formats. You must set:
- `base_dataset_folder`: path to the original REAL-Colon dataset  
- `output_folder`: path where the formatted dataset will be written
   
To replicate the full pipeline used in this repository, **the predefined train/validation/test split should be left unchanged**.

### Code Origins & Credits
The dataset conversion script `export_yolo_coco_format.py` in this folder is adapted from the following repository: https://github.com/cosmoimd/yolov7.

## 3. Perform sampling of positive/negative frames
Edit the input/output paths inside `sampling.py`:

- `TRAIN_JSON_PATH`, `VAL_JSON_PATH`, `TEST_JSON_PATH`
- `NEG_PER_VIDEO_TRAIN`, `NEG_PER_VIDEO_VAL`, `POS_PER_VIDEO_VAL`

Then run:
``` python
python sampling.py   
```
This script produces the sampled JSON files used for building the filtered YOLO and COCO datasets.

## 4. Final YOLO and RT-DETR Dataset
### 4.1 YOLO Dataset
After running the sampling script, run:
```python
python build_yolo_sampled_dataset.py
```
Before running, update the following line in the script: 
```python
BASE_DIR = Path("/path/to/your/dir")
```
BASE_DIR must contain the following folders:
```
split/        # Output of export_yolo_coco_format.py  (full dataset)
dataset/      # Output of sampling_real_colon.py      (filtered JSONs)
```

The script will automatically create:
```
final_yolo/
   train/
      images/
      labels/
      train_ann.json
   val/
      images/
      labels/
      val_ann.json
```
      
This produces the final YOLO-ready dataset containing only the sampled images and labels.

### 4.2 RT-DETR Dataset
To generate RT-DETR-compatible JSONs, run:
```
python build_rtdetr_sampled_dataset.py
```

Before running, update the following line in the script: 
```python
BASE_DIR = Path("/path/to/your/dir")
```
BASE_DIR must contain the following folders:
```
dataset/      # Output of sampling_real_colon.py      (filtered JSONs)
```
The script will automatically create:
```
RTDETR/
 train_ann.json
 val_ann.json
 test_ann.json
```
These JSON files are directly compatible with RT-DETR training.
