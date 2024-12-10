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

Datasets are stored in the `data/` folder or can be accessed directly from [this link](https://github.com/yogeshjangir16/Line-Detection-Using-Yolo/tree/657fabdf4e64dd8196123761cea3ee0fadb2957d/Dataset/data).

## **Line Detection Architecture with OpenCV**

The following architecture is used for detecting lines using OpenCV.

<div align="center">
  <img src="https://github.com/yogeshjangir16/Line-Detection-Using-Yolo/blob/7374d635fb28b54e17acfcdd6ded3a9deba51acd/mdData/architecture_cv.png" alt="Line Detection Architecture" width="500"/>
  <p><em>Figure 1: Line Detection Architecture with OpenCV</em></p>
</div>

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
   git clone https://github.com/yogeshjangir16/Line-Detection-Using-YOLO.git
   cd Line-Detection-Using-YOLO

2. Install Dependencies
   Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
3. Prepare the Dataset
- Annotate the dataset using LabelImg.
- Upload and preprocess the dataset on Roboflow, then download it in the YOLO format.

4. Train the Model
   Use the [Roboflow](https://roboflow.com) for training:

## **Results**
