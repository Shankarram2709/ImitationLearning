# ImitationLearning
This repository contains a deep learning model for predicting future vehicle waypoints using detected objects, lane information, and IMU data. The model employs attention mechanisms and GRU-based decoding to handle variable-length inputs and generate 4 waypoints (0.5s apart).

## Features
- **Input Modalities**: Objects, lanes, and IMU data.
- **Attention Pooling**: Handles variable-length object/lane detections.
- **GRU Decoder**: Generates sequential waypoints.
- **Normalization**: Per-modality feature normalization.
- **Validation Metrics**: ADE (Average Displacement Error), FDE (Final Displacement Error).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Shankarram2709/ImitationLearning.git
2. Train the model:
   Make sure to keep all training data and labels present inside the same folder for parsing
   ```bash
   python main.py --config "Path to configs folder containing yaml" --data-dir "Path to the dataset containing objects, lanes, imu and 
   waypoint information"
   
