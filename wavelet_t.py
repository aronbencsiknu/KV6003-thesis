import numpy as np
import matplotlib.pyplot as plt

# Generate time values
t = np.linspace(0, 2*np.pi, 512, endpoint=False)

# Generate a time-series with high and low frequencies
data = 0.5 * np.sin(2 * t)

# Add some noise
noise = 0.5 * np.random.randn(len(t))
data += noise

# Plot the original time-series data
plt.plot(t, data)
plt.title("Original Time-series Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

import pywt

# Apply DWT transform to the time-series data
cA, cD = pywt.dwt(data, 'db2')

# Remove the low-frequency components by setting appropriate coefficients to zero
lmbd = 0.5

for i in range(len(cD)):
    if abs(cD[i]) > lmbd:
        cD[i] = 0

for i in range(len(cA)):
    if abs(cA[i]) > lmbd:
        cA[i] = 0

print(cD)

# Reconstruct the denoised time-series data using inverse DWT
denoised_data = pywt.idwt(cA, cD, 'db2')

# Plot the denoised time-series data
plt.plot(t, denoised_data)
plt.title("Denoised Time-series Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
