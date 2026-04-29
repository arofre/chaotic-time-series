import matplotlib.pyplot as plt
import numpy as np

class Reservoir:
    def __init__(self, neurons = 500):
        self.input_weights = np.random.normal(0, np.sqrt(0.002), (neurons, 3))
        self.weights = np.random.normal(0, np.sqrt(0.004), (neurons, neurons))
        self.output_weights = np.random.normal(0, np.sqrt(0.002), (3, neurons))
        self.state = np.zeros(neurons)

    def step(self, input_vector):
        self.state = np.tanh(
            np.dot(self.input_weights, input_vector)
            + np.dot(self.weights, self.state)
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

        self.output_weights = (
            targets.T
            @ states
            @ np.linalg.inv(states.T @ states + alpha * I)
        )

np.random.seed(42)

training_data = np.loadtxt("training-set.csv", delimiter=",")
test_data = np.loadtxt("test-set.csv", delimiter=",")

X, Y, Z = training_data[0], training_data[1], training_data[2]
X_test, Y_test, Z_test = test_data[0], test_data[1], test_data[2]

inputs = np.column_stack((X[:-1], Y[:-1], Z[:-1]))
targets = np.column_stack((X[1:], Y[1:], Z[1:]))


reservoir = Reservoir()
reservoir.ridge_regression(inputs, targets, alpha=0.01)

warmup_steps = 20

for input_vector in np.column_stack(
    (X_test[:warmup_steps], Y_test[:warmup_steps], Z_test[:warmup_steps])):
    reservoir.step(input_vector)

last_output = np.array([
    X_test[warmup_steps - 1],
    Y_test[warmup_steps - 1],
    Z_test[warmup_steps - 1]
])

num_forecast_steps = min(500, len(X_test) - warmup_steps)
predictions = []

for _ in range(num_forecast_steps):
    reservoir.step(last_output)
    last_output = reservoir.output()
    predictions.append(last_output)

predictions = np.array(predictions)

predictions_x = predictions[:, 0]
predictions_y = predictions[:, 1]
predictions_z = predictions[:, 2]

np.savetxt("prediction.csv", predictions_y, delimiter=",")

true_x = X_test[warmup_steps:warmup_steps + num_forecast_steps]
true_y = Y_test[warmup_steps:warmup_steps + num_forecast_steps]
true_z = Z_test[warmup_steps:warmup_steps + num_forecast_steps]

true = np.column_stack((true_x, true_y, true_z))

mse = np.mean((predictions - true) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - true))

y_mse = np.mean((predictions_y - true_y) ** 2)
y_rmse = np.sqrt(y_mse)
y_mae = np.mean(np.abs(predictions_y - true_y))

trajectory_range = np.max(true) - np.min(true)
nrmse = rmse / trajectory_range

y_range = np.max(true_y) - np.min(true_y)
y_nrmse = y_rmse / y_range

threshold = 0.2 * y_range
y_errors = np.abs(predictions_y - true_y)

within_threshold = y_errors < threshold
forecast_horizon = np.argmax(~within_threshold)

if forecast_horizon == 0 and within_threshold[0]:
    forecast_horizon = len(within_threshold)

print("Overall MSE:", mse)
print("Overall RMSE:", rmse)
print("Overall MAE:", mae)
print("Overall NRMSE:", nrmse)

print("Y MSE:", y_mse)
print("Y RMSE:", y_rmse)
print("Y MAE:", y_mae)
print("Y NRMSE:", y_nrmse)

print("Forecast horizon:", forecast_horizon, "steps")
print("Error threshold:", threshold)

plt.figure(figsize=(10, 5))
plt.plot(true_y, label="True Y")
plt.plot(predictions_y, label="Predicted Y")
plt.xlabel("Time step")
plt.ylabel("Y value")
plt.title("Reservoir Computing Forecast on Lorenz Time Series")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_errors)
plt.axhline(threshold, linestyle="--", label="20% range threshold")
plt.xlabel("Time step")
plt.ylabel("Absolute Y error")
plt.title("Forecast Error Over Time")
plt.legend()
plt.tight_layout()
plt.show()
