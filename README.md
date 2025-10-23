# 🚗 Traffic Vehicle Detection & Counting System

A comprehensive YOLOv8-based vehicle detection and counting system trained on Indian traffic data. This project implements a two-stage training approach to achieve high accuracy in vehicle classification and counting.

## 🎯 **Project Overview**

This system detects and counts 8 different types of vehicles commonly found in Indian traffic:
- **Auto** (Auto-rickshaw)
- **Bus** 
- **Car**
- **LCV** (Light Commercial Vehicle)
- **Motorcycle**
- **Multiaxle** (Multi-axle truck)
- **Tractor**
- **Truck**

## 🏆 **Key Achievements**

- **Stage 1 Model**: 98.03% mAP50 accuracy
- **Stage 2 Model**: Enhanced fine-tuned model with improved truck/bus classification
- **Real-time Processing**: Capable of processing video streams
- **Vehicle Tracking**: Persistent tracking across frames
- **Counting System**: Accurate vehicle counting with classification

## 📊 **Model Performance**

### Stage 1 Model (`yolov8m_stage1_smart`)
- **mAP50**: 98.03%
- **Precision**: 98.1%
- **Recall**: 98.0%
- **Training Data**: 8,219 images
- **Classes**: 8 vehicle types

### Stage 2 Model (`yolov8m_stage2_improved`)
- **Fine-tuned** on additional truck/bus examples
- **Enhanced** classification accuracy for challenging cases
- **Improved** performance on real-world traffic scenarios

## 🚀 **Quick Start**

### Prerequisites
```bash
Python 3.8+
PyTorch
Ultralytics YOLO
OpenCV
```

### Installation
1. Clone this repository:
```bash
git clone https://github.com/vietnguyen755/highway-detection-vehicle-
cd highway-detection-vehicle-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Download required files** (required):
   - **Trained Models**: Download from [Google Drive Link] or contact team lead
   - **Dataset Images**: Download from [Dataset Link] and place in `dataset/train/images/` and `new_finetunedata/train/images/`
   - **Test Video**: Watch demo on [YouTube Link] or download from [Video Link]

### Usage

#### Run Detection on Video
```bash
python main.py
```

#### Run with Custom Model
```python
from main import VehicleCounter

# Initialize with custom model
counter = VehicleCounter(
    model_path="runs/detect/yolov8m_stage2_improved/weights/best.pt",
    video_source="your_video.mp4"
)

# Process video
counter.process_video()
```

#### Batch Processing
```bash
# Use the provided batch file
test_improved_model.bat
```

## 📁 **Project Structure**

```
highway-detection-vehicle/
├── main.py                          # Main application
├── requirements.txt                 # Python dependencies
├── dataset/                         # Original training data
│   ├── data.yaml                   # Dataset configuration
│   └── train/
│       ├── images/                 # Download images separately
│       └── labels/                 # Label files included
├── new_finetunedata/               # Fine-tuning dataset
│   ├── data.yaml                   # Dataset configuration
│   └── train/
│       ├── images/                 # Download images separately
│       └── labels/                 # Label files included
├── runs/                           # Model outputs (download separately)
│   └── detect/
│       ├── yolov8m_stage1_smart/   # Stage 1 model files
│       └── yolov8m_stage2_improved/ # Stage 2 model files
├── PROJECT_REPORT.md              # Detailed project documentation
├── test_improved_model.bat        # Testing script
└── README.md                      # This file
```

**Note**: Large files (images, videos, models) are excluded from this repository. Download them separately as instructed above.

## 🔧 **Technical Details**

### Model Architecture
- **Base Model**: YOLOv8m (Medium)
- **Input Size**: 640x640 pixels
- **Classes**: 8 vehicle types
- **Training**: Two-stage approach

### Training Process
1. **Stage 1**: Train on large dataset (8,219 images)
2. **Stage 2**: Fine-tune on specific examples (92 images)
3. **Validation**: Test on real traffic videos

### Key Features
- **Real-time Detection**: Processes video frames in real-time
- **Vehicle Tracking**: Maintains vehicle IDs across frames
- **Classification Correction**: Built-in correction for common misclassifications
- **Counting System**: Tracks vehicles entering/exiting the frame
- **Visualization**: Draws bounding boxes and labels

## 📈 **Results**

The system successfully detects and classifies vehicles with high accuracy:

- **Cars**: 99.2% accuracy
- **Trucks**: 97.8% accuracy  
- **Buses**: 98.5% accuracy
- **Motorcycles**: 99.1% accuracy
- **Auto-rickshaws**: 98.7% accuracy

## 🎥 **Demo Video**

The `detection_output_improved.mp4` file contains a demonstration of the system processing real traffic footage, showing:
- Vehicle detection with bounding boxes
- Classification labels
- Vehicle counting
- Tracking across frames

## 📚 **Documentation**

- **PROJECT_REPORT.md**: Comprehensive project documentation
- **results.txt**: Latest performance results
- **data.yaml**: Dataset configuration files

## 🤝 **Contributing**

This project is open for contributions! Areas for improvement:
- Additional vehicle types
- Better tracking algorithms
- Mobile deployment
- Real-time web interface

## 📄 **License**

This project is available under the MIT License.

## 🙏 **Acknowledgments**

- **Ultralytics** for the YOLOv8 framework
- **PyTorch** for deep learning capabilities
- **OpenCV** for computer vision processing

## 📞 **Contact**

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This model is specifically trained on Indian traffic patterns and may require fine-tuning for other geographical regions.
