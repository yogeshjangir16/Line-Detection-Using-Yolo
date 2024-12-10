# **Line Detection in Historical Indic Manuscripts Using YOLO**

This project focuses on developing a robust line detection model for historical Indic manuscripts using YOLO (You Only Look Once) deep learning frameworks. The model effectively handles complex and noisy manuscripts, enabling accurate text segmentation across diverse scripts such as Khmer, Sundanese, and Balinese.


## **Features**
- **Line Detection**: Achieves precise segmentation of text lines in historical manuscripts.
- **Multilingual Support**: Supports Khmer, Sundanese, and Balinese datasets.
- **Noise Robustness**: Handles noisy and degraded images effectively.
- **YOLO Models**: Comparison of YOLOv11 and YOLO NAS for optimal results.



## **Datasets**
The project leverages historical Indic manuscripts from the following datasets:
1. **Khmer**
2. **Sundanese**
3. **Balinese**

Datasets are stored in the `data/` folder or can be accessed directly from [this link](https://github.com/yogeshjangir16/Line-Detection-Using-Yolo/tree/657fabdf4e64dd8196123761cea3ee0fadb2957d/Dataset).



## **Annotation Tools**
For annotation of the datasets:
- **LabelImg**: Used for creating bounding boxes for text lines.
  - Install LabelImg using:
    ```bash
    pip install labelImg
    ```
  - Run LabelImg:
    ```bash
    labelImg
    ```
- **Roboflow**: Used to preprocess and format data for YOLO training. Visit [Roboflow](https://roboflow.com) to manage your dataset.



## **Steps to Setup**

Follow these steps to set up the project:

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/username/Line-Detection-Using-YOLO.git
   cd Line-Detection-Using-YOLO
