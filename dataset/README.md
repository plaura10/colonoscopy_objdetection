# Dataset Formatting 
This repository provides scripts to format the REAL-Colon dataset to the format for the models YOLOv7, YOLOv11 (YOLO format) and RT-DETR (COCO format). 

## Dataset Downloads
First, download the REAL-Colon dataset at the link: https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866. 
After you have this dataset, unzip all compressed files. In total, you will have 120 folders, since the dataset contains 60 videos. Each video has two linked folders, one containing the frames in jpg; the other one containing the annotations in xml format. The file also contains a lesion_info.csv file and video_info.csv file, which contain information about the lesion and the video contained in this dataset. 

## Installation
Clone the repository and follow these steps to set up the necessary environment:
1. **Install Python**: ensure that python 3.8 or later is installed on your system.
2. **Set up a virtual environment** (optional but recommended):
   ```python
   python -m venv dataset
   source dataset/bin/activate  # On Windows use `dataset\Scripts\activate`
   ```
3. **Install dependencies** using the requirements file
   ``` python
   pip install -r requirements.txt
   ```
   
## Dataset Conversion and Split
This allows you to convert the REAL-Colon dataset into YOLO and COCO formats. You set the base_dataset_folder (path to the original REAL-Colon dataset) and the output_folder (where the formatted dataset will be written).
To replicate the entire pipeline the predefined train/validation/test split should be left unchanged.

### Code Origins & Credits
The dataset conversion script `export_yolo_coco_format.py` in this folder is adapted from the following repository: https://github.com/cosmoimd/yolov7.
