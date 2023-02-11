import numpy as np


def sig(x):
    return 1 / (1 + np.exp(-x))


training_data = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]])
# [inputs, cases]
training_inputs = training_data[:, 0:3]
# [outputs, cases]
training_outputs = training_data[:, 3:]

# np.random.seed(1)

# [inputs, 1st HL count]
W1 = 2 * np.random.random((3, 5)) - 1
# [1st HL count, outputs]
W2 = 2 * np.random.random((5, 1)) - 1

print("Initial synaptic weights:")
print(W2)
print(W1)
print("Initial test output:")
V1 = np.dot(np.array([1, 1, 0]), W1)
F1 = sig(V1)
V2 = np.dot(F1, W2)
F2 = sig(V2)
print(F2)

for i in range(20000):
    V1 = np.dot(training_inputs, W1)
    F1 = sig(V1)
    V2 = np.dot(F1, W2)
    F2 = sig(V2)

    E2 = training_outputs - F2
    G2 = E2 * F2 * (1 - F2)

    E1 = np.dot(G2, W2.T)
    G1 = E1 * F1 * (1 - F1)

    A2 = np.dot(F1.T, G2)
    A1 = np.dot(training_inputs.T, G1)
    W2 += A2
    W1 += A1


print("Final synaptic weights:")
print(W2)
print(W1)
print("Final test output:")
V1 = np.dot(np.array([1, 1, 0]), W1)
F1 = sig(V1)
V2 = np.dot(F1, W2)
F2 = sig(V2)
print(F2)
