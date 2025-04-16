# Autonomous Vehicle Lane & Object Detection System

##  Project Overview

This project develops a lane and object detection system for self-driving cars using deep learning. It can identify:

- Lane markings (solid/dashed lines)
- Vehicles (cars, trucks, motorcycles)
- Pedestrians
- Traffic signs

## Key Features

# Faster R-CNN with ResNet-50 backbone  
# BDD100K dataset (100,000+ driving clips)  
# Works in various weather/lighting conditions  
# 63% detection accuracy in initial tests  
# Easy-to-use Jupyter Notebook interface  

## Installation

1. Clone the repository:

git clone https://github.com/waliur957/autonomous-driving.git
cd autonomous-driving


2. Install requirements:

pip install -r requirements.txt

##  Quick Start

1. Open the Jupyter notebook:

jupyter notebook Autonomous_Driving_Detection.ipynb


2. Run the cells sequentially to:
- Load the model
- Process sample images
- View detection results

## Project Structure

autonomous-driving-detection/
├── models/               
├── data/                 
├── utils/                
├── Autonomous_Driving_Detection.ipynb  
├── requirements.txt      
└── README.md         


## Performance

| Metric    | Score |
|-----------|-------|
| Accuracy  | 63%   |
| Precision | 10%   |
| Recall    | 10%   |
| F1 Score  | 10%   |

*Performance on BDD100K validation set

