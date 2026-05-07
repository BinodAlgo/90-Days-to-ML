"""
# Forward Pass Neural Network Architecture
Setup
Inputs (X): A batch of 3 samples, where each sample has 4 features (Shape: 3x4).
Weights (W): A weight matrix for a layer with 2 neurons (Shape: 4x2).
Biases (b): One bias per neuron (Shape: 2,).

"""

import numpy as np

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
W = np.random.randn(4, 2)
b = np.array([0.5, -0.5])

# Calculate Weighted Sum 
Z = X.dot(W) + b 
print(f"Weighted Sum (Z): \n{Z}\n") 


# Activation 
# The most common activation function is ReLU (Rectified Linear Unit) 
# It introduces non-linearity into the network, allowing it to learn complex patterns
A = np.maximum(0, Z)
print(f"Activations (A): \n{A}\n")

# 