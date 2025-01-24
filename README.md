# Potato Disease Classifier

## Overview
The **Potato Disease Classifier** is a deep learning-based image classification project designed to identify diseases in potato leaves. Using a Convolutional Neural Network (CNN), the model classifies images of potato leaves into three categories:

1. **Potato___Early_blight**
2. **Potato___Late_blight**
3. **Potato___healthy**

By applying **data augmentation techniques**, such as flipping the images, the model effectively generates more samples to improve its performance. After training for **50 epochs**, the model achieved remarkable accuracy:

- **Training Accuracy**: 99.73%
- **Test Accuracy**: 99.27%

---

## Features
- **Image Classification**: Classifies potato leaf images into one of three categories.
- **Data Augmentation**: Enhances the dataset using image flipping for better generalization.
- **Deep Learning Model**: Employs a CNN with multiple layers for accurate classification.
- **High Accuracy**: Achieved over 99% accuracy on both training and test datasets.

---

## Dataset
The dataset contains images of potato leaves categorized into the following classes:
1. **Potato___Early_blight**: Leaves affected by early blight disease.
2. **Potato___Late_blight**: Leaves affected by late blight disease.
3. **Potato___healthy**: Healthy potato leaves with no visible disease symptoms.

Data augmentation techniques were applied to increase the diversity of the training data and improve model robustness.

---

## Model Architecture
The model is built using a **Convolutional Neural Network (CNN)**, which includes:
- Convolutional layers for feature extraction.
- Pooling layers for dimensionality reduction.
- Fully connected layers for classification.

### Key Parameters:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50

---

## Results
The model was trained and tested on the dataset, achieving the following results:
- **Training Accuracy**: 99.73%
- **Test Accuracy**: 99.27%

The high accuracy demonstrates the effectiveness of the CNN model and the data augmentation approach.

---

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Aksh-acc/plant-disease-classifier/potato-disease-classifier.git
   cd potato-disease-classifier
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Training Script**:
   Train the model on the dataset.
   ```bash
   python train.py
   ```

4. **Make Predictions**:
   Use the trained model to classify new images of potato leaves.
   ```bash
   python predict.py --image path_to_image
   ```

---

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **NumPy**
- **Matplotlib**
- **OpenCV**

---

## Future Enhancements
- Add support for real-time image classification using a webcam or mobile app.
- Extend the classifier to include more plant diseases.
- Integrate Grad-CAM to visualize which parts of the leaf images contribute most to predictions.

---

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

