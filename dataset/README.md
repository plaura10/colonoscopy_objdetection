# Dataset Formatting 
This repository provides scripts to format the REAL-Colon dataset to the format for the models YOLOv7, YOLOv11 and RT-DETR. 

## Dataset Downloads
First, downoaled the REAL-Colon dataset at the link: https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866. 
After you have this dataset, unzip all compressed files. At the end you have 120 folders, since the dataset contains 60 video, each video has two linked folders, one containing the frames in jpsg; the other one contaiing the annotations in xml format. The file also contains a lesion_info.csv file and video_info.csv file, which contain information about the lesion and the video contained in this dataset. 

## Installation
Clone the repository and follow these steps to seut up the necessary environmente:
1. **Install Python**: ensure that python 3.8 or later is installed on your system.
2. **Set up a virtual environment** (optional but reccomended):
   '''python
   python -m venv dataset
   source dataset/bin/activate  # On Windows use `dataset\Scripts\activate`
   '''
3. **Install dependencies using the requirements file
   '''python
   pip install -r requirements.txt
   '''

# Dataset convertion and split  
I performed this split 


