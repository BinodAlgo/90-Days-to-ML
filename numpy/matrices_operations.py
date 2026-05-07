# 1. Indexing and Slicing(Extracting Data)
import numpy as np 
matrix_A = np.array([
    [2,4,5],
    [5,9,2],
    [3,1,8]
])
print(f"Matrix A =\n{matrix_A}\n")
# Print first element
print(f"First element: {matrix_A[0,0]}\n")
# Print second row
print(f"Second row: {matrix_A[1,:]}\n")
# Print third column
print(f"Third column: {matrix_A[:,2]}\n")
# Print specific element (row 2, column 1)
print(f"Element at (2,1): {matrix_A[1,0]}\n")

# 2. Broadcasting ---> Adding scalar to a matrix
scalar = 3 
result = matrix_A + scalar
print(f"Result of adding scalar to matrix A:\n{result}\n")

# Matrix Multiplication 
matrix_B = np.array([
    [1,2],
    [3,4],
    [5,6]
])
result = np.dot(matrix_A, matrix_B)
print(f"Result of matrix multiplication of A and B:\n{result}\n")


# Aggregations and Axes 
'''
Calculating statistics like Sum, Mean, Max or Min. In Machine learning, we us this for 
calculating average error(loss) across a batch of data, or finding which output neuron 
has the highest probability score. Axes are used to specify the dimension along which
the operation should be performed. 
'''
matrix_C = np.eye(3)
print(f"Matrix C =\n{matrix_C}\n")
# Sum 
print(f"Sum of all elements: {np.sum(matrix_C)}\n")
# Mean
print(f"Mean of all elements: {np.mean(matrix_C)}\n")
# Max
print(f"Max of all elements: {np.max(matrix_C)}\n")
# Min
print(f"Min of all elements: {np.min(matrix_C)}\n")
# Axis  
print(f"Mean of elements along axis 1 (rows): {np.mean(matrix_C,axis=1)}\n")
print(f"Mean of elements along axis 0 (columns): {np.mean(matrix_C, axis=0)}")

# Matrix Transposition 
matrix_C_Transpose = np.transpose(matrix_C)
print(f"Matrix C Transpose = \n{matrix_C_Transpose}\n")


# Stacking and Concatenation
''' 
Gluing matrices together. In neural network this is often used to join multiple 
batches of data like matrix of image data and another matrix of audio data, and then feed
them into a model. For this, we need to stick them together. We can do this using 
np.vstack, np.hstack, np.dstack, np.concatenate.

'''
matrix_2x2_a = np.array([
    [2,2],
    [3,4]
])

# vertical stack 
matrix_2x2_b = np.array([
    [5,0],
    [6,2]
])

result_vstack = np.vstack((matrix_2x2_a, matrix_2x2_b))
result_hstack = np.hstack((matrix_2x2_a, matrix_2x2_b))

print(f"Vertical Stack result:\n{result_vstack}\n")
print(f"Horizontal Stack result:\n{result_hstack}\n")