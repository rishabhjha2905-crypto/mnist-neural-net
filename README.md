# MNIST Handwritten Digit Classifier
 
A feedforward neural network built from scratch in PyTorch that recognizes handwritten digits (0-9) from the MNIST dataset with 96.67% accuracy.
 
## Overview
 
MNIST is the classic benchmark for neural networks — 60,000 training images and 10,000 test images of handwritten digits, each 28x28 pixels. This project builds a simple 3-layer feedforward network, trains it using backpropagation and the Adam optimizer, and evaluates it on unseen test data.
 
The misclassified images reveal that the network's remaining errors are largely on genuinely ambiguous handwriting — cases where even humans would struggle — indicating the model has learned meaningful digit representations rather than just memorizing patterns.
 
## Architecture
 
```
Input (784) → Hidden Layer 1 (128) → ReLU → Hidden Layer 2 (64) → ReLU → Output (10)
```
 
- **Input layer:** 784 neurons (one per pixel in a 28x28 image)
- **Hidden layer 1:** 128 neurons with ReLU activation
- **Hidden layer 2:** 64 neurons with ReLU activation
- **Output layer:** 10 neurons (one per digit 0-9)
- **Total parameters:** ~109,000 learnable weights and biases
## Training
 
- **Loss function:** Cross Entropy Loss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 5
- **Batch size:** 64

| Epoch | Loss |
|-------|------|
| 1 | 0.3984 |
| 2 | 0.1915 |
| 3 | 0.1413 |
| 4 | 0.1134 |
| 5 | 0.0949 |
 
## Results
 
**Test Accuracy: 96.67%**
 
The network's mistakes are concentrated on ambiguous handwriting — for example, a 9 written with sharp angles being mistaken for a 4, or a European-style 2 being mistaken for a 7. These are cases that would challenge human readers as well.
 
## Tech Stack
 
- Python
- PyTorch
- torchvision
- matplotlib
## How to Run
 
1. Clone the repo
2. Install dependencies:
   ```
   pip install torch torchvision matplotlib
   ```
3. Open `mnist.ipynb` in VS Code or Jupyter and run all cells
4. The MNIST dataset downloads automatically on first run
## Using the Saved Model
 
A pre-trained model is included (`mnist_model.pth`). To load and use it:
 
```python
import torch
model = NeuralNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()
```
 
## Project Structure
 
```
mnist-neural-net/
├── mnist.ipynb         # Main notebook
├── mnist_model.pth     # Pre-trained model weights
├── .gitignore          # Excludes data folder
└── README.md
```