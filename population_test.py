import numpy as np
import matplotlib.pyplot as plt

class CUBANeuron:
    def __init__(self, tau=0.01, g=0.2, Vth=0.1):
        self.tau = tau
        self.g = g
        self.Vth = Vth
        self.V = 0
        self.spike = 0
    
    def update(self, Iext, dt):
        # Update membrane potential using Euler method
        dV = (-self.V + self.g*Iext) / self.tau
        self.V += dV*dt
        
        # Generate output spike if membrane potential reaches threshold
        if self.V >= self.Vth:
            self.spike = 1
            self.V = 0
        else:
            self.spike = 0

def simulate_neuron(Iext_mean=1, Iext_std=0.01, dt=0.001, T=0.1):
    # Define time vector and input current
    t = np.arange(0, T, dt)
    Iext = np.random.normal(Iext_mean, Iext_std, size=len(t))
    
    # Create neuron object and simulate
    neuron = CUBANeuron()
    V = np.zeros_like(t)
    spikes = np.zeros_like(t)
    for i in range(len(t)):
        neuron.update(Iext[i], dt)
        V[i] = neuron.V
        spikes[i] = neuron.spike
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(t, V, label='Membrane potential')
    plt.plot(t, spikes, label='Output spikes')
    plt.plot([t[0], t[-1]], [neuron.Vth, neuron.Vth], 'k--', label='Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Potential or Spikes')
    plt.legend()
    plt.show()

    print(spikes)

# Call the function to simulate the neuron
simulate_neuron()
simulate_neuron(Iext_mean=0.5)
simulate_neuron(Iext_mean=0.1)
simulate_neuron(Iext_mean=0.09)