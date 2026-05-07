import numpy as np 
# Manual Creation 
# We use nested list to create a matrix 
A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

# Structural Creation 
# We often don't know the vlaues, we just know the shape or dimension 
B = np.zeros((3,4)) # a matrix of shape 3x4 with all zeros
C = np.ones((2,2)) # a matrix of shape 2X2 with all ones 
D = np.eye(3) # a identity matrix of size 3x3 


# Random Initialization
# In ML, we often start weights with random numbers 
weights = np.random.rand(3,2) # a matrix of shape 3X2 with random values between 0 and 1
print(f"A = {A}\n")
print(f"B = {B}\n")
print(f"C = {C}\n")
print(f"D = {D}\n")
print(f"weights = {weights}\n")

# In ML, we often get data in wrong shape or dimension, so we need to reshape the data
array_1D = np.array([1,2,3,4,5,6]) 
array_2D = array_1D.reshape((2,3))
array_3D = array_1D.reshape((1,2,3))
print(f"array_3D = {array_3D}\n")