# Solar Panel Fault and Efficiency Detection

An end-to-end machine learning system designed to monitor solar panel health. This project uses a combination of Gradient Boosting (XGBoost) for tabular anomaly detection and Convolutional Neural Networks (CNN) for identifying physical faults (like snow cover or dust) from images.

## 🚀 Overview
Solar energy efficiency drops significantly due to environmental factors and hardware faults. This system provides:
- **Anomaly Detection:** Identifies irregular power output patterns using XGBoost.
- **Image Classification:** Categorizes panel conditions (Clean, Dusty, Snow-Covered, Bird-Droppings).
- **Efficiency Prediction:** Estimates current output efficiency based on environmental telemetry.

## 📊 Dataset
To keep the repository lightweight, the raw image dataset is hosted externally.
- [Download the Solar Image Dataset here](https://www.kaggle.com/datasets/viandre/solar-panel-images)

> **Note:** After downloading, place the images in the `data/Solar panel Image/` directory to run the training notebooks.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow / Keras (CNN Architecture)
- **Machine Learning:** XGBoost, Scikit-Learn
- **Data Handling:** Pandas, NumPy
- **Deployment:** Python (Predict System API)

## 📁 Project Structure
```plaintext
├── data/
│   └── Solar_Panel_5000_Days_Advanced_Dataset.xlsx  # Tabular telemetry
├── notebooks/
│   ├── image classification and analysis.ipynb      # CNN Training logic
│   └── training.ipynb                               # XGBoost & Tabular logic
├── models/
│   ├── efficiency_model.pkl                         # Saved XGBoost weights
│   ├── xgb_anomaly_model.pkl                        # Anomaly detector
│   └── best_model.h5                                # CNN Model
└── predict_system.py                                # Integrated inference script

Installation
Clone the repository:
git clone [https://github.com/Pratikshukla13/Solar-Panel-Fault-and-Efficiency-Detection.git](https://github.com/Pratikshukla13/Solar-Panel-Fault-and-Efficiency-Detection.git)
cd Solar-Panel-Fault-and-Efficiency-Detection

Install dependencies:
pip install -r requirements.txt

