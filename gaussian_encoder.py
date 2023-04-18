import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def create_population(center, sigma, n_fields, field_size):
    population = np.zeros((n_fields, field_size))
    start = center - sigma * (n_fields-1)/2
    end = center + sigma * (n_fields-1)/2
    x = np.linspace(0, 1, field_size)
    for i in range(n_fields):
        mu = start + i * sigma
        field = gaussian(x, mu, sigma)
        field /= np.max(field)  # Normalize the receptive field
        population[i] = field
    return population

# Example usage
populations = []
center = 0.5
sigma = 0.1
n_fields = 10
num_populations = 5
field_size = 50
for i in range(num_populations):
    population = create_population(center, sigma, n_fields, field_size)
    populations.append(population)

# Plot the populations
fig, axs = plt.subplots(5, 1, figsize=(6, 8))
for i, population in enumerate(populations):
    for j, receptive_field in enumerate(population):
        axs[i].plot(receptive_field)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_ylabel(f"Population {i+1}")
plt.show()
