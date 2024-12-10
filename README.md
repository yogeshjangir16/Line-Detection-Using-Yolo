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

### **Line Detection Performance with OpenCV**
The performance of line detection using OpenCV is demonstrated below with an evaluation of clear and noisy text images.

<div align="center">
  <img src="https://github.com/yogeshjangir16/Line-Detection-Using-Yolo/blob/47d9211098d2a840a5c7cb1839b26cc75dd76135/mdData/output_cv.png" alt="Line Detection using OpenCV" width="700"/>
  <p><em>Figure 1: Line Detection on Clear vs Noisy Images using OpenCV</em></p>
</div>

### **Evaluation Metrics for YOLO Models**
Below are the evaluation results for YOLOv11 and YOLO-NAS models, showing precision and recall across different datasets.

<div align="center">
  <table>
    <tr>
      <th>Dataset</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
    <tr>
      <td>Khmer</td>
      <td>0.979</td>
      <td>0.974</td>
    </tr>
    <tr>
      <td>Sundanese</td>
      <td>0.986</td>
      <td>0.995</td>
    </tr>
    <tr>
      <td>Balinese</td>
      <td>0.957</td>
      <td>0.916</td>
    </tr>
  </table>
  <p><em>Figure 2: Evaluation Metrics for YOLOv11</em></p>
</div>

<div align="center">
  <table>
    <tr>
      <th>Dataset</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
    <tr>
      <td>Khmer</td>
      <td>0.992</td>
      <td>0.975</td>
    </tr>
    <tr>
      <td>Sundanese</td>
      <td>0.958</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Balinese</td>
      <td>1.0</td>
      <td>0.952</td>
    </tr>
  </table>
  <p><em>Figure 3: Evaluation Metrics for YOLO-NAS</em></p>
</div>

### **Training Graphs of YOLO Models**
The training graphs illustrate the learning curves of the models on the Khmer dataset.

<div align="center">
  <img src="https://github.com/yogeshjangir16/Line-Detection-Using-Yolo/blob/47d9211098d2a840a5c7cb1839b26cc75dd76135/mdData/traingraph_yolov11_Khamer.png" alt="Training Graph YOLOv11 Khmer" width="700"/>
  <p><em>Figure 4: Training Graph of YOLOv11 on Khmer Dataset</em></p>
</div>

<div align="center">
  **Training Graph for YOLO-NAS on Khmer Dataset**
  <img src="https://github.com/yogeshjangir16/Line-Detection-Using-Yolo/blob/47d9211098d2a840a5c7cb1839b26cc75dd76135/mdData/traingraph_yolovNAs_Khamer.png" alt="Training Graph YOLO-NAS Khmer" width="700"/>
  <p><em>Figure 5: Training Graph of YOLO-NAS on Khmer Dataset</em></p>
</div>

### **Performance Metrics Comparison**
The performance of YOLOv11 and YOLO-NAS was further evaluated using a performance matrix, highlighting the IoU and Hausdorff distance (HD) metrics.

<div align="center">
  <table>
    <tr>
      <th>Dataset</th>
      <th>IoU</th>
      <th>HD</th>
      <th>Avg HD</th>
    </tr>
    <tr>
      <td>Khmer</td>
      <td>0.96</td>
      <td>2.67</td>
      <td>2.35</td>
    </tr>
    <tr>
      <td>Sundanese</td>
      <td>0.9667</td>
      <td>2.66</td>
      <td>2.35</td>
    </tr>
    <tr>
      <td>Balinese</td>
      <td>0.97</td>
      <td>2.67</td>
      <td>2.33</td>
    </tr>
  </table>
  <p><em>Figure 6: Performance Matrix of YOLOv11</em></p>
</div>

<div align="center">
  <table>
    <tr>
      <th>Dataset</th>
      <th>IoU</th>
      <th>HD</th>
      <th>Avg HD</th>
    </tr>
    <tr>
      <td>Khmer</td>
      <td>0.95</td>
      <td>2.67</td>
      <td>2.356</td>
    </tr>
    <tr>
      <td>Sundanese</td>
      <td>0.9667</td>
      <td>2.66</td>
      <td>2.35</td>
    </tr>
    <tr>
      <td>Balinese</td>
      <td>0.98</td>
      <td>2.65</td>
      <td>2.34</td>
    </tr>
  </table>
  <p><em>Figure 7: Performance Matrix of YOLO-NAS</em></p>
</div>

## **Conclusion**
The project demonstrated significant advancements in line detection within historical Indic manuscripts using YOLO models. YOLOv11 and YOLO-NAS were evaluated across multiple datasets, achieving promising results in terms of precision, recall, and overall model performance. 

Key takeaways:
- YOLO-NAS outperformed YOLOv11 in terms of accuracy, achieving near-perfect scores on precision and recall for datasets like Balinese and Sundanese.
- The training graphs indicated that YOLO-NAS had faster convergence and better stability.
- The performance metrics, including IoU and Hausdorff distances, validated the superior line localization of YOLO-NAS.

Future improvements can be made by incorporating better preprocessing techniques and fine-tuning the models for specific dataset characteristics to further enhance the detection accuracy and robustness.

## **Contact**
For any questions or further information, please feel free to reach out:

**Email:** [jangir.7@iitj.ac.in](mailto:jangir.7@iitj.ac.in)

