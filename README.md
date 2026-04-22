# MNIST Handwritten Digit Classifier
 
An interactive web app powered by a Convolutional Neural Network (CNN) trained in PyTorch that recognizes handwritten digits (0-9) with 99.24% accuracy. Draw a digit on the canvas and the model predicts it in real time.
 
## Demo
 
[Live App on Streamlit](#) ← add your link here after deploying
 
## Overview
 
This project builds and compares two neural network architectures on the MNIST dataset:
 
1. **Feedforward Neural Network** — a simple 3-layer network that flattens the image into 784 pixels. Achieves 96.67% accuracy.
2. **Convolutional Neural Network (CNN)** — uses convolutional layers to detect spatial features like edges and curves directly in the image. Achieves 99.24% accuracy.
The CNN is then deployed as an interactive Streamlit web app where users can draw digits on a canvas and get real-time predictions with confidence scores.
 
## Architecture
 
### CNN (Final Model)
```
Input (1x28x28) → Conv2d(32) → ReLU → MaxPool → Conv2d(64) → ReLU → MaxPool → Flatten → Linear(128) → Dropout(0.25) → Linear(10)
```
 
### Feedforward Network (Baseline)
```
Input (784) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(10)
```
 
## Results
 
| Model | Test Accuracy |
|-------|--------------|
| Feedforward Neural Network | 96.67% |
| CNN | 99.24% |
 
The CNN's remaining errors are concentrated on genuinely ambiguous handwriting — cases where even humans would struggle — indicating the model has learned meaningful digit representations.
 
## Training
 
- **Loss function:** Cross Entropy Loss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 10 (CNN), 5 (Feedforward)
- **Batch size:** 64

| Epoch | CNN Loss |
|-------|----------|
| 1 | 0.1740 |
| 2 | 0.0574 |
| 3 | 0.0409 |
| 5 | 0.0267 |
| 10 | 0.0124 |
 
## Tech Stack
 
- Python
- PyTorch, torchvision
- Streamlit
- streamlit-drawable-canvas
- matplotlib
## How to Run Locally
 
1. Clone the repo
2. Install dependencies:
   ```
   pip install torch torchvision streamlit streamlit-drawable-canvas matplotlib
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Or open `mnist.ipynb` to see the full training process
## Project Structure
 
```
mnist-neural-net/
├── mnist.ipynb            # Training notebook
├── app.py                 # Streamlit web app
├── mnist_model.pth        # Saved feedforward model
├── mnist_cnn_model.pth    # Saved CNN model
├── .gitignore             # Excludes data folder
└── README.md
```
 