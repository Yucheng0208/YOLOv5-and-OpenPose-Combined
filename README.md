# Computer Vision-Based Bus Stop Passenger Detection System

This project aims to develop a computer vision-based bus stop passenger detection system by integrating YOLOv5 for object detection and OpenPose for pose estimation. The system can detect passengers waiting at bus stops, recognize whether they are using wheelchairs or other assistive devices, and provide voice alerts to the driver.

## Abstract

Passenger transport is one of the most common ways of commuting in Taiwan. It plays an important role in the transportation system due to its large number of stations, dense frequency, and cheap transportation. Due to the unfriendly transportation environment and a large number of passengers, a blind spot of passenger transportation exists, which leads to traffic accidents at the station. We research to make the "Bus Stop Passenger Detection System". Taking the object detection of "Wheelchairs" into consideration, it is more convenient to assist the disabled to find the passenger transportation system, which makes Taiwan's transportation system more convenient.

Published in: 2023 IEEE 3rd International Conference on Electronic Communications, Internet of Things and Big Data (ICEIB)

Date of Conference: 14-16 April 2023

Date Added to IEEE Xplore: 07 July 2023

ISBN Information:

DOI: 10.1109/ICEIB57887.2023.10169919

Publisher: IEEE

Conference Location: Taichung, Taiwan

## Features

- **Object Detection**: Utilizes YOLOv5 to detect passengers, wheelchairs, and other objects.
- **Pose Estimation**: Employs OpenPose to identify passenger postures such as sitting, standing, and hand-raising.
- **Hardware Integration**: Implements Bluetooth detection and pairing via Raspberry Pi to determine the proximity between the bus and the stop, and uses OpenCV to capture images for analysis.

## Prerequisites

Before starting, ensure your system meets the following requirements:

- **Operating System**: Ubuntu 20.04 or a compatible Linux distribution.
- **Python Version**: Python 3.8 or higher.
- **Dependencies**: Refer to the `requirements.txt` file for a complete list of dependencies.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Yucheng0208/YOLOv5-and-OpenPose-Combined.git
   cd YOLOv5-and-OpenPose-Combined
   ```

2. **Create and Activate a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights**:

   - **YOLOv5 Model**: Follow the [YOLOv5 official guide](https://github.com/ultralytics/yolov5) to download pretrained models.
   - **OpenPose Model**: Refer to the [OpenPose official guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to obtain the necessary model files.

5. **Configure Paths**:

   Ensure that the paths to the downloaded model files are correctly set in the project's configuration files so the system can load the models properly.

## Usage

1. **Start the System**:

   ```bash
   python main.py
   ```

2. **System Workflow**:

   - The system uses Raspberry Pi's Bluetooth functionality to detect the distance between the bus and the stop.
   - Upon detecting the bus approaching the stop, it captures images using OpenCV.
   - The images are first processed by YOLOv5 for object detection, identifying passengers and wheelchairs.
   Subsequently, OpenPose analyzes the images for pose estimation, recognizing passenger postures such as sitting, standing, and hand-raising.
   - Based on the detection results, the system provides voice alerts to the driver, especially concerning passengers requiring assistance.

## Academic Citation

If you use this project in your research, please cite the following paper:

**Title**: Based-on Computer Vision Applications for Bus Stop Passenger Detection System

**Authors**: Yu-Cheng Chang, Hua-Wen Tsai, Chao-Yi Huang, Zong-Rong Wu

**Conference**: 2023 IEEE International Conference on Consumer Electronics-Taiwan (ICCE-TW)

**DOI**: [10.1109/ICEIB57887.2023.10169919](https://doi.org/10.1109/ICEIB57887.2023.10169919)

**IEEE Xplore**: [https://ieeexplore.ieee.org/document/10169919](https://ieeexplore.ieee.org/document/10169919)

**BibTeX**:

```BibTeX
@inproceedings{chang2023based,
  title={Based-on Computer Vision Applications for Bus Stop Passenger Detection System},
  author={Chang, Yu-Cheng and Tsai, Hua-Wen and Huang, Chao-Yi and Wu, Zong-Rong},
  booktitle={2023 IEEE 3rd International Conference on Electronic Communications, Internet of Things and Big Data (ICEIB)},
  pages={152-154},
  year={2023},
  organization={IEEE},
  doi={10.1109/ICEIB57887.2023.10169919}
}
```

## Notes

- Please ensure your devices are correctly set up with the required dependencies and model files.
- For practical applications, consider fine-tuning the models based on your specific environment to enhance detection accuracy.
- If you encounter any issues, refer to the project's [GitHub page](https://github.com/Yucheng0208/YOLOv5-and-OpenPose-Combined) for more information or to submit an issue.

We hope this project improves passenger detection and safety at bus stops.
