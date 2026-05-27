# ✍️ Handwritten Digit Recognizer (Deep Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikamsudarshan/handwritten_digit_recognizer/blob/main/Handwritten_Digit_Recognizer.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4)

An end-to-end Deep Learning project that classifies handwritten digits in real-time. This project demonstrates the entire machine learning pipeline: from data ingestion and model training using an Artificial Neural Network (ANN) to deploying an interactive web UI within a Jupyter environment.

> **Note:** Want to try it yourself? Click the **Open in Colab** badge above, hit `Runtime > Run All`, and scroll to the bottom to draw a digit!

## 🚀 The Project Overview

While training a model on the MNIST dataset is a standard introductory exercise, deploying it to process real-world, user-generated drawings introduces a classic problem: **Data Distribution Shift**. 

This project bridges that gap by implementing a custom Computer Vision preprocessing pipeline (using OpenCV) to dynamically format user inputs to match the strict mathematical distribution the neural network expects.

### Key Features
* **Custom ANN Architecture:** Built using TensorFlow/Keras with Dense layers, ReLU activation, and Dropout for regularization.
* **Interactive UI:** Leverages Gradio to provide a digital canvas for real-time inference.
* **Automated Image Preprocessing:**
  * Inverts RGBA canvas strokes to grayscale (White text on Black background).
  * Implements dynamic bounding-box cropping to auto-center digits, regardless of where they are drawn on the canvas.
  * Applies thresholding and padding to mimic MNIST framing.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Processing:** NumPy, Matplotlib
* **User Interface:** Gradio

## 📸 Preview
*`![App Demo](/handwritten_digit_recognizer/assets/demo_screenshot.png)`*

## 🧠 The Preprocessing Pipeline Explained

To ensure the model achieves high accuracy on live drawings, the raw input from the Gradio canvas undergoes several transformations before inference:

1. **Grayscale & Inversion:** The model expects black backgrounds with white digits. The canvas defaults to white. We invert the bits (`cv2.bitwise_not`) to correct this.
2. **Auto-Centering:** The MNIST dataset features tightly cropped, centered digits. If a user draws a small digit in the corner of the canvas, the model will fail. We use `cv2.findNonZero` and `cv2.boundingRect` to find the drawing, crop out the empty space, and add uniform padding.
3. **Downscaling:** The image is cleanly resized to the $28 \times 28$ pixel tensor expected by the neural network's input layer.
4. **Normalization:** Pixel values are scaled from $[0, 255]$ down to a $[0.0, 1.0]$ float range.

## 💻 How to Run Locally

If you prefer to run this on your local machine instead of Google Colab:

1. Clone this repository:

```bash
git clone https://github.com/nikamsudarshan/handwritten_digit_recognizer.git
```
   
2. Install the required dependencies:
 ```bash
pip install -r requirements.txt
```

3.Open the Jupyter Notebook and run the cells sequentially:
```bash
jupyter notebook Handwritten_Digit_Recognizer.ipynb
```
Author: Sudarshan Nikam
