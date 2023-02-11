import numpy as np


def sig(x):
    return 1 / (1 + np.exp(-x))


training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Initial synaptic weights:")
print(synaptic_weights)
print("Initial test output:")
print(sig(np.dot(np.array([1, 1, 0]), synaptic_weights)))

for i in range(20000):
    outputs = sig(np.dot(training_inputs, synaptic_weights))
    err = training_outputs - outputs
    adjustments = np.dot(training_inputs.T, err * outputs * (1 - outputs))
    synaptic_weights += adjustments

print("Final synaptic weights:")
print(synaptic_weights)
print("Final test output:")
print(sig(np.dot(np.array([1, 1, 0]), synaptic_weights)))
