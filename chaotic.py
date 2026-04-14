import matplotlib.pyplot as plt
import numpy as np
import csv

class Reservoir:
    def __init__(self):
        self.input_weights = np.random.normal(0, np.sqrt(0.002), (500, 3))
        self.weights = np.random.normal(0, np.sqrt(0.004), (500, 500))
        self.output_weights = np.random.normal(0, np.sqrt(0.002), (3, 500))
        self.state = np.zeros(500)

    def step(self, input_vector):
        self.state = np.tanh(
            np.dot(self.input_weights, input_vector) + np.dot(self.weights, self.state)
        )

    def output(self):
        return np.dot(self.output_weights, self.state)

    def ridge_regression(self, inputs, targets, alpha):
        states = []
        for input_vector in inputs:
            self.step(input_vector)
            states.append(self.state.copy())
        states = np.array(states)

        I = np.identity(states.shape[1])
        self.output_weights = targets.T @ states @ np.linalg.inv(states.T @ states + alpha * I)

training_data = []
test_data = []

training_data = np.loadtxt('training-set.csv', delimiter=',')
test_data = np.loadtxt('test-set.csv', delimiter=',')

X, Y, Z = training_data[0], training_data[1], training_data[2]
X_test, Y_test, Z_test = test_data[0], test_data[1], test_data[2]

inputs = np.column_stack((X[:-1], Y[:-1], Z[:-1]))
targets = np.column_stack((X[1:], Y[1:], Z[1:]))

reservoir = Reservoir()
reservoir.ridge_regression(inputs, targets, alpha=0.01)

predictions = []

for input_vector in np.column_stack((X_test, Y_test, Z_test)):
    reservoir.step(input_vector)
    last_output = reservoir.output()

predictions_y = []
for _ in range(500):
    reservoir.step(last_output)
    last_output = reservoir.output()
    predictions_y.append(last_output[1]) 

predictions_y = np.array(predictions_y)

np.savetxt("prediction.csv", predictions_y, delimiter=",")

