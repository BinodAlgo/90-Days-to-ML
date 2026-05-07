'''
Your Task: Build the Neuron
Define Inputs: Create a 1D array (vector) with the values [1.2, 0.7, 2.1]. 
Define Weights: Create a 1D array with random values 
(use np.random.rand(3)).
Define Bias: Set a scalar value, e.g., bias = 0.5.
Calculate the Output:Use np.dot() to multiply inputs and weights.Add the bias.

'''
import numpy as np
inputs = np.array([1.2,0.7,2.1])
weights = np.random.rand(3)
bias = 0.5 
output = np.dot(inputs, weights) + bias
print(f"Input {inputs}")
print(f"Weights {weights}")
print(f"Bias {bias}")
print(f"Output {output}")
