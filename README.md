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

## **Results and Discussion**

### **Line Detection Performance with OpenCV**
The performance of line detection using OpenCV is demonstrated below with an evaluation of clear and noisy text images.

<div align="center">
  <img src="path/to/output_cv.png" alt="Line Detection using OpenCV" width="700"/>
  <p><em>Figure 1: Line Detection on Clear vs Noisy Images using OpenCV</em></p>
</div>

### **Evaluation Metrics for YOLO Models**
Below are the evaluation results for YOLOv11 and YOLO-NAS models, showing precision and recall across different datasets.

**YOLOv11 Evaluation Metrics**

| **Dataset** | **Precision** | **Recall** |
|-------------|---------------|------------|
| Khmer       | 0.979         | 0.974      |
| Sundanese   | 0.986         | 0.995      |
| Balinese    | 0.957         | 0.916      |

<div align="center">
  <img src="path/to/eval_yolov11.png" alt="YOLOv11 Evaluation Metrics" width="700"/>
  <p><em>Figure 2: Evaluation Metrics for YOLOv11</em></p>
</div>

**YOLO-NAS Evaluation Metrics**

| **Dataset** | **Precision** | **Recall** |
|-------------|---------------|------------|
| Khmer       | 0.992         | 0.975      |
| Sundanese   | 0.958         | 1.0        |
| Balinese    | 1.0           | 0.952      |

<div align="center">
  <img src="path/to/eval_yoloNAS.png" alt="YOLO-NAS Evaluation Metrics" width="700"/>
  <p><em>Figure 3: Evaluation Metrics for YOLO-NAS</em></p>
</div>

### **Training Graphs of YOLO Models**
The training graphs illustrate the learning curves of the models on the Khmer dataset.

**Training Graph for YOLOv11 on Khmer Dataset**

<div align="center">
  <img src="path/to/train_graph_yolov11_khmer.png" alt="Training Graph YOLOv11 Khmer" width="700"/>
  <p><em>Figure 4: Training Graph of YOLOv11 on Khmer Dataset</em></p>
</div>

**Training Graph for YOLO-NAS on Khmer Dataset**

<div align="center">
  <img src="path/to/train_graph_yoloNAS_khmer.png" alt="Training Graph YOLO-NAS Khmer" width="700"/>
  <p><em>Figure 5: Training Graph of YOLO-NAS on Khmer Dataset</em></p>
</div>

### **Performance Metrics Comparison**
The performance of YOLOv11 and YOLO-NAS was further evaluated using a performance matrix, highlighting the IoU and Hausdorff distance (HD) metrics.

**Performance Metrics for YOLOv11**

| **Dataset** | **IoU**  | **HD**  | **Avg HD** |
|-------------|----------|---------|------------|
| Khmer       | 0.96     | 2.67    | 2.35       |
| Sundanese   | 0.9667   | 2.66    | 2.35       |
| Balinese    | 0.97     | 2.67    | 2.33       |

<div align="center">
  <img src="path/to/performance_matrix_yolov11.png" alt="Performance Matrix YOLOv11" width="700"/>
  <p><em>Figure 6: Performance Matrix of YOLOv11</em></p>
</div>

**Performance Metrics for YOLO-NAS**

| **Dataset** | **IoU**  | **HD**  | **Avg HD** |
|-------------|----------|---------|------------|
| Khmer       | 0.95     | 2.67    | 2.356      |
| Sundanese   | 0.9667   | 2.66    | 2.35       |
| Balinese    | 0.98     | 2.65    | 2.34       |

<div align="center">
  <img src="path/to/performance_matrix_yoloNAS.png" alt="Performance Matrix YOLO-NAS" width="700"/>
  <p><em>Figure 7: Performance Matrix of YOLO-NAS</em></p>
</div>

