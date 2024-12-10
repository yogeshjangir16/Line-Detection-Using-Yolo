# Line Detection in Historical Indic Manuscripts

This project explores the application of advanced deep learning techniques, specifically YOLOv11 and YOLO-NAS, to detect text lines in historical Indic manuscripts. The focus is on handling challenges such as noise, overlapping characters, and skewed text in diverse datasets, including Khmer, Sundanese, and Balinese manuscripts.

---

## **Project Overview**
Historical Indic manuscripts often contain degraded text due to age, noise, and complex layouts. Traditional OCR methods struggle to accurately segment text lines under such conditions. This project aims to enhance line detection using YOLO-based models, improving the digitization, transcription, and preservation of these invaluable documents.

---

## **Key Features**
- **Robust Line Detection**: Handles noisy and skewed manuscript images with overlapping text lines.
- **Multi-Dataset Performance**: Tested on Khmer, Sundanese, and Balinese manuscript datasets.
- **Comparative Analysis**: Demonstrates YOLO-NAS's superior performance over YOLOv11.

---

## **Figures and Tables**
### **Figures**
- **Training Graphs**: YOLOv11 and YOLO-NAS training performance across datasets.
- **Segmentation Results**: Comparison of clear vs. noisy image performance.
- Stored in the `figures/` folder.

### **Tables**
- Evaluation metrics such as Precision, Recall, IoU, and Hausdorff Distance.
- Performance comparison of YOLOv11 and YOLO-NAS across datasets.
- Stored in the `tables/` folder.

---

## **Datasets**
The project leverages historical Indic manuscripts from the following datasets:
1. **Khmer**
2. **Sundanese**
3. **Balinese**

Datasets are stored in the `data/` folder or can be accessed [https://github.com/yogeshjangir16/Line-Detection-Using-Yolo/tree/657fabdf4e64dd8196123761cea3ee0fadb2957d/Dataset](#).

---

## **Installation**
### **Prerequisites**
- Python 3.8 or higher
- GPU with CUDA support (optional, but recommended)

### **Steps to Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/Line-Detection-Using-YOLO.git
   cd Line-Detection-Using-YOLO

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
