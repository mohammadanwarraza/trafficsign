# 🛑 AI-Based Traffic Signal Detection System

This project implements a real-time traffic signal detection system using deep learning and computer vision techniques. A Convolutional Neural Network (CNN) is trained to classify traffic signals from image data and deployed to detect signals live via webcam input.

## 🔍 Features

- Image preprocessing (grayscale, histogram equalization, normalization)  
- Data augmentation using ImageDataGenerator  
- CNN model built with TensorFlow/Keras  
- Real-time video stream classification using OpenCV  
- Model evaluation with precision, recall, F1-score, and accuracy metrics  
- Pretrained model loading/saving with Pickle  
- Live traffic signal prediction with overlayed results  

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Scikit-learn  
- Pandas  
- Pickle  

## 📁 Dataset

Custom traffic signal images organized by class in folders (stored in `myData/`).  
Dataset source: https://www.kaggle.com/datasets/abdallahwagih/traffic-signs-dataset/data  
Labels for each class are provided in `labels.csv`.

## 🚀 How to Run

1. Clone the repository  
2. Add your dataset in the `myData/` folder and `labels.csv`  
3. Run the Python script to train or load the model  
4. Launch webcam-based real-time detection  

## 📊 Results

- Achieved ~97% test accuracy in classifying traffic signals across various categories and conditions.

## 📌 Future Improvements

- Use larger and more diverse datasets  
- Optimize model for embedded systems (e.g., Raspberry Pi)  
- Expand to detect additional road signs or dynamic traffic lights  

## 🙏 Thank You!!! 😊
