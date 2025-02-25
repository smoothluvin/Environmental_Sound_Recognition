import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 500         # Number of samples
mu = 0.01       # Learning rate (step size)
M = 4           # Filter order (number of taps)

# Input signal (random noise)
x = np.random.randn(N)

# Unknown system (FIR filter we want to "learn")
true_weights = np.array([0.5, -0.3, 0.2, -0.1])  # Unknown system coefficients
d = np.convolve(x, true_weights, mode='same')  # Desired signal

# Initialize LMS filter
w = np.zeros(M)  # Initial filter weights
y = np.zeros(N)  # LMS output
e = np.zeros(N)  # Error signal

# Training loop
for n in range(M, N):
    x_n = x[n:n-M:-1]  # Recent M samples
    y[n] = np.dot(w, x_n)  # Filter output
    e[n] = d[n] - y[n]  # Error signal
    w += mu * e[n] * x_n  # Update weights

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(d, label="Desired Signal")
plt.plot(y, label="LMS Output", linestyle="dashed")
plt.plot(e, label="Error", linestyle="dotted")
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("LMS Filter Training")
plt.show()

print("Learned Weights:", w)
print("True Weights:   ", true_weights)