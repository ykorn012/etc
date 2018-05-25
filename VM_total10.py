import numpy as np


Tgt = np.array([0, 50])
yk = np.array([2.9,  0.2])
uk = np.array([0.6,  0.3])
A = np.array([[0.5, -0.2], [0.25, 0.15]])
L1 = 0.55 * np.identity(2)
L2 = 0.75 * np.identity(2)
I = np.identity(2)

Dk1 = np.array([1.6, 0.2])
Qk1 = np.array([1.2, 0.5])

Dk = (yk - uk.dot(A)).dot(L1) + Dk1.dot((I - L1))

Qk = (yk - uk.dot(A) - Dk1).dot(L2) + Qk1.dot(I - L2)

uk1 = (Tgt - Dk - Qk).dot(np.linalg.inv(A))

print(uk1)

print(uk1.dot(A))

print(A)
B = np.transpose(A)
print(B)
print(A * B)
print(np.linalg.inv(A.dot(B)))

print(A.dot(np.linalg.inv(A)))








