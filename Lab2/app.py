import numpy as np
from utils_to_complete_etudiant import inverseHomogeneousMatrix, multiplyHomogeneousMatrix, skew

# Test inverseHomogeneousMatrix
T = np.array([[1, 0, 0, 3],
              [0, 1, 0, 4],
              [0, 0, 1, 5],
              [0, 0, 0, 1]])

inv_T = inverseHomogeneousMatrix(T)
print("Inverse of Homogeneous Matrix:")
print(inv_T)

# Test multiplyHomogeneousMatrix
T1 = np.array([[1, 0, 0, 2],
               [0, 1, 0, 3],
               [0, 0, 1, 4],
               [0, 0, 0, 1]])

T2 = np.array([[1, 0, 0, 5],
               [0, 1, 0, 6],
               [0, 0, 1, 7],
               [0, 0, 0, 1]])

T12 = multiplyHomogeneousMatrix(T1, T2)
print("\nProduct of Two Homogeneous Matrices:")
print(T12)

# Test skew
t = np.array([1, 2, 3])
sk_matrix = skew(t)
print("\nSkew Symmetric Matrix:")
print(sk_matrix)
