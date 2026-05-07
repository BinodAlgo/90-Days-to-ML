# Beside basic operations on matrices, numpy also provides way to filter data based on conditions
import numpy as np
temperatures = np.array([21,25,30,18,22,27,19])
# create a mask
hot_days_mask = temperatures > 25
print(f"Boolean mask for hot days: {hot_days_mask}\n")
hot_days = temperatures[hot_days_mask]
print(f"Hot days (above 25 degrees Celsius): {hot_days}")

# 2. Vectorization and Broadcasting
'''
Vectorization: Performing operations on entire arrays at once without using loops.
Broadcasting: Performing operations on arrays of different shapes.
Rule of Broadcasting:
1. Two dimensions are either equal, or
2. One of them is 1
How Broadcasting Works ? 
(M, N) + (M, 1) --> (M, N)
(M, N) + (1, N) --> (M, N)
(M, N) + (M, N) --> (M, N)
'''



# Vectorization

matrix_A = np.array([
    [1,2],
    [3,4]
])
vector_b = np.array([1,2])
result = matrix_A + vector_b
print(f"Result of vectorization: {result}")

# Broadcasting
# adding scalar to a matrix
scalar = 5 
result_scalar = matrix_A + scalar
print(f"Result of adding scalar to matrix A: {result_scalar}")

'''
Vectorization: "I don't use a for loop, just do A + B so the computer does it super fast in a batch."
Broadcasting: "Oh, A and B are different sizes? No problem, I'll stretch the smaller one for you so A + B still works."

'''

matrix_B = np.array([1,3])

print(f"\nResult of broadcasting: {matrix_A + matrix_B}")