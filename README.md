# Alina-Zahra---Machine-Learning---Month-1---Task-2-MNIST-Digit-Recognition

> An end-to-end Convolutional Neural Network (CNN) web application built for real-time handwritten digit recognition and classification. Developed as part of the Machine Learning Internship Program at **Arch Technologies**.

---

## 🔗 Live Demo
<img width="1919" height="903" alt="image" src="https://github.com/user-attachments/assets/fde8a77e-8c20-42a7-aebe-25c250636a2a" />

* **Live Application:** [Streamlit Live App](https://mnist-digit-recognition-app.streamlit.app/)

---

## 📌 Problem Statement & Motivation
Handwritten digit recognition serves as a foundational benchmark in computer vision and deep learning. This project implements a robust Convolutional Neural Network (CNN) trained on the MNIST dataset of handwritten digits (0 to 9) to accurately preprocess images, train a classification model, and evaluate its performance in real-time through an interactive web interface.

---

## ✨ Key Features & Architecture
* **CNN Classification Engine:** Powered by a deep learning convolutional model optimized for pixel image feature extraction.
* **Interactive Drawing Canvas:** Features a responsive built-in drawing board allowing users to sketch digits live.
* **Image Upload Support:** Enables users to upload external handwritten digit images (PNG/JPG) for immediate inference.
* **Live Probability Distribution:** Visualizes confidence scores and class probability breakdown across all digits from 0 to 9.
* **Cyberpunk Glassmorphism UI:** Designed with a sleek, modern dark-mode interface built using custom CSS and Streamlit components.

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Deep Learning & Frameworks:** TensorFlow, Keras, NumPy, Pillow (PIL)
* **Frontend & UI:** Streamlit, Streamlit-Drawable-Canvas, Custom CSS
* **Deployment & Version Control:** Streamlit Community Cloud, GitHub

---

## 🔄 Project Architecture & Workflow
1. **Input Acquisition:** User draws a digit on the interactive canvas or uploads an image file.
2. **Image Preprocessing:** The input is resized to 28x28 pixels, converted to grayscale, and pixel values are normalized.
3. **Model Inference:** The trained CNN model evaluates the processed array to predict the correct digit class and calculate confidence metrics.
4. **Result Presentation:** The dashboard renders the predicted digit alongside a complete breakdown of class probabilities.

---

## ⚙️ Installation & Local Setup Guide

Follow these steps to run the project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AlinaZahraHub/Alina-Zahra---Machine-Learning---Month-1---Task-2-MNIST-Digit-Recognition.git](https://github.com/AlinaZahraHub/Alina-Zahra---Machine-Learning---Month-1---Task-2-MNIST-Digit-Recognition.git)
   cd Alina-Zahra---Machine-Learning---Month-1---Task-2-MNIST-Digit-Recognition

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the application:**
```bash
streamlit run app.py

```



---

## 📂 Project Structure

```text
mnist-digit-recognition/
│
├── app.py                      # Main Streamlit application interface and logic
├── MNIST DIGIT RECOGNITION.ipynb # Jupyter Notebook containing training and evaluation
├── mnist_digit_model.h5        # Trained CNN classification model
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```

---

## 👩‍💻 Author & Acknowledgement

* **Name:** Alina Zahra
* **Internship Program:** Machine Learning Internship & Training Program (August – September 2026)
* **Organization:** Arch Technologies
* **GitHub:** [AlinaZahraHub](https://github.com/AlinaZahraHub)
* **LinkedIn:** [Alina Zahra](https://www.linkedin.com/in/alina-zahra12/)

```

```
