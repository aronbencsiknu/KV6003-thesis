import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch

class CUBANeuron(nn.Module):
    def __init__(self, tau, g, Vth, dt, t, exponent):
        super(CUBANeuron, self).__init__()
        self.tau = nn.Parameter(torch.tensor(tau))
        self.g = nn.Parameter(torch.tensor(g))
        self.Vth =nn.Parameter(torch.tensor(Vth))
        self.exponent = nn.Parameter(torch.tensor(exponent))
        """self.tau = tau
        self.g = g
        self.Vth = Vth
        self.exponent = exponent"""
        self.V = 0
        self.spike = 0
        
        self.fired = False
        self.time_to_fire = int(t/dt)
    
    def forward(self, Iext, dt, index):
        # Update membrane potential using Euler method
        #dV = (-self.V + self.g*Iext) / self.tau

        if not self.fired:
            EPSILON = 1e-8
            #dV = -(1/self.tau)*self.V + ((1/(abs(self.g-Iext)+EPSILON))**self.exponent)
            dV = -(1/self.tau)*self.V + Iext * self.g
            self.V += dV*dt
        
        # Generate output spike if membrane potential reaches threshold
        if self.V >= self.Vth and not self.fired:
            self.spike = 1
            self.V = 0
            self.fired = True
            self.time_to_fire = index
        else:
            self.spike = 0

        return self.spike


class CUBAPopulation():
    def __init__(self, population_size, tau, g, Vth, dt, t, exponent):
        self.population_size = population_size
        self.tau = tau
        self.g = g
        self.Vth = Vth
        self.exponent = exponent

        self.neurons = []

        for i in range(0, self.population_size):
            self.neurons.append(CUBANeuron(tau, g+(i/2)*2, Vth, dt, t, exponent))

    def forward(self, Iext, dt, index):
        spikes_at_t = []
        for i in range(0, self.population_size):
            spikes_at_t.append(self.neurons[i](Iext, dt, index))

        return spikes_at_t


class CUBALayer():
    def __init__(self, sample_size, population_size, tau=0.1, g=-5.0, Vth=0.01, dt=0.001, t=0.1, exponent=2.0):
        self.sample_size = sample_size
        self.population_size = population_size
        self.tau = tau
        self.g = g
        self.Vth = Vth

        self.populations = []

        for i in range(self.sample_size):
            self.populations.append(CUBAPopulation(population_size, tau, g, Vth, dt, t, exponent))
            

    def forward(self, Iext, dt, index):
        population_spikes_at_t = []
        for i in range(0, self.sample_size):
            population_spikes_at_t.append(self.populations[i].forward(Iext, dt, index))

        return population_spikes_at_t

def simulate_neuron(spike_times, Iext_mean=1, dt=0.001, T=0.1):
    # Define time vector and input current
    t = np.arange(0, T, dt)
    Iext = np.full(len(t), Iext_mean)
    
    # Create neuron object and simulate
    cuba_layer = CUBALayer(1, 10)
    V = np.zeros_like(t)
    #spikes = np.zeros_like(t)
    spikes = []
    for i in range(len(t)):
        spikes.append(cuba_layer.forward(Iext[i], dt, i))
        #V[i] = neuron.V
    index =  0
    for i in range(len(cuba_layer.populations[0].neurons)):
        spike_times[i].append(cuba_layer.populations[0].neurons[i].time_to_fire)
    #spike_times.append(cuba_layer.populations[0].neurons[0].time_to_fire)

    return spike_times

# Call the function to simulate the neuron
current_mean = -0.5
spike_times1 = [[] for _ in range(10)]
spike_times2 = []

x_labels = []

for i in range(100):
    spike_times1 = simulate_neuron(spike_times1, Iext_mean=current_mean)

    current_mean += 0.01
    x_labels.append(round(current_mean,3))


print(np.asarray(spike_times1).shape)

for i in range(10):
    plt.plot(spike_times1[i])
    
plt.gca().invert_yaxis()
tick_locs = np.arange(0, len(x_labels), 10)
plt.xticks(tick_locs, x_labels[::10])
plt.show()