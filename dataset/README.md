# Dataset Formatting 
This repository provides scripts to format the REAL-Colon dataset to the format for the models YOLOv7, YOLOv11 (YOLO format) and RT-DETR (COCO format). 

## 1. Dataset Downloads
First, download the REAL-Colon dataset at the link: https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866. 
After you have this dataset, unzip all compressed files. In total, you will have 120 folders, since the dataset contains 60 videos. Each video has two linked folders, one containing the frames in jpg; the other one containing the annotations in xml format. The file also contains a lesion_info.csv file and video_info.csv file, which contain information about the lesion and the video contained in this dataset. 

   
## 2. Dataset Conversion and Split
This allows you to convert the REAL-Colon dataset into YOLO and COCO formats. You set the base_dataset_folder (path to the original REAL-Colon dataset) and the output_folder (where the formatted dataset will be written).
To replicate the entire pipeline the predefined train/validation/test split should be left unchanged.

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

## 4. Final YOLO and COCO dataset
### 4.1 Build YOLO Dataset
After running the sampling script, run:
```python
python build_yolo_sampled_dataset.py
```
Before running, update the following line in the script: 
```python
BASE_DIR = Path("/path/to/your/dir")
```
BASE_DIR must contain the following folders:
split/        # Output of export_yolo_coco_format.py  (full dataset)
dataset/      # Output of sampling_real_colon.py      (filtered JSONs)

The script will automatically create:
final_yolo/
   train/images/
   train/labels/
   val/images/
   val/labels/

