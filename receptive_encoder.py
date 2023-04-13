import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import random
import math

class CUBANeuron(nn.Module):
    def __init__(self, tau, g, Vth, dt, T):
        super(CUBANeuron, self).__init__()
        self.tau = tau
        self.g = g
        self.Vth = Vth
        self.V = 0
        self.spike = 0

        self.voltages = []
        
        self.fired = False
        self.time_to_fire = int(T/dt)
    
    def forward(self, Iext, dt, index):
        
        # Update membrane potential using Euler method
        self.V = math.exp(-dt/self.tau) * self.V + self.g * Iext
        self.voltages.append(self.V)
        
        
        # Generate output spike if membrane potential reaches threshold
        
        if self.V >= self.Vth and not self.fired:
            self.spike = 1
            #self.V = 0
            self.fired = True
            self.time_to_fire = index
        else:
            self.spike = 0

        return self.spike


class CUBAPopulation():
    def __init__(self, population_size, tau=0.1, g=-5.0, Vth=1, dt=0.001, T=0.1):
        self.population_size = population_size
        tau = 0.9
        self.tau = tau
        self.g = g
        self.Vth = Vth
        self.dt = dt
        self.T = T

        self.neurons = []

        self.gains = self.generate_gain(population_size, -0.3, 0.3, tau, dt, T, Vth)
        self.create_neurons()       
    
    def create_neurons(self):
        for i in range(0, self.population_size):
            self.neurons.append(CUBANeuron(tau=self.tau, g=self.gains[i], Vth=self.Vth, dt=self.dt, T=self.T))

    def generate_gain(self, count, intercept_low, intercept_high, tau, dt=0.001, T=0.1, Vth=1):
        gain = []

        for i in range(count):
            intercept_current = random.uniform(intercept_low, intercept_high)
            
            if intercept_current > 0:
                multiplier = 1
            else:
                multiplier = -1
            
            g_min = multiplier * 1000

            last_voltage = Vth + 2
            counter = 0
            while last_voltage > Vth+0.1:
                counter += 1
                g_min = g_min - ((last_voltage - Vth)/200) * multiplier
                test_neuron = CUBANeuron(tau, g_min, Vth, dt, T)
                for j in range(0, int(T/dt)):
                    test_neuron.forward(intercept_current, dt, j)

                #print(test_neuron.voltages[-1])
                #print((last_voltage - Vth)**2)
                last_voltage = test_neuron.voltages[-1]

            print(counter, "_",i,":",g_min, " last voltage", last_voltage, " Vth", Vth)
            gain.append(g_min)

        print("DONE")  
        return gain
    
    def reset_neurons(self):
        self.neurons = []
        self.create_neurons()

    def forward(self, Iext, dt, index):
        spikes_at_t = []
        for i in range(0, self.population_size):
            spikes_at_t.append(self.neurons[i](Iext, dt, index))

        return spikes_at_t


class CUBALayer():
    def __init__(self, sample_size, population_size, tau=0.1, g=-5.0, Vth=1, dt=0.001, T=0.1):
        self.sample_size = sample_size
        self.population_size = population_size
        self.tau = tau
        self.g = g
        self.Vth = Vth

        self.populations = []

        for i in range(self.sample_size):
            self.populations.append(CUBAPopulation(population_size, tau, g, Vth, dt, T))
            

    def forward(self, Iext, dt, index):
        population_spikes_at_t = []
        for i in range(0, self.sample_size):
            population_spikes_at_t.append(self.populations[i].forward(Iext, dt, index))

        return population_spikes_at_t
    
    def reset_neurons(self):
        for i in range(self.sample_size):
            self.populations[i].reset_neurons()


"""def simulate_neuron(spike_times, cuba_layer, Iext_mean=1, dt=0.001, T=0.1):
    # Define time vector and input current
    t = np.arange(0, T, dt)
    Iext = np.full(len(t), Iext_mean)
    #cuba_layer = CUBALayer(1, 10)
    #spikes = np.zeros_like(t)
    spikes = []
    for i in range(len(t)):
        spikes.append(cuba_layer.forward(Iext[i], dt, i))

    for i in range(len(cuba_layer.populations[0].neurons)):
        spike_times[i].append(cuba_layer.populations[0].neurons[i].time_to_fire)

    cuba_layer.reset_neurons()


    return spike_times


# Call the function to simulate the neuron
current_mean = -1
spike_times = [[] for _ in range(10)]

x_labels = []
cuba_layer = CUBALayer(1, 10)
for i in range(100):
    spike_times = simulate_neuron(spike_times, cuba_layer, Iext_mean=current_mean)
    current_mean += 0.02
    x_labels.append(round(current_mean,3))

print(np.asarray(spike_times).shape)

for i in range(10):
    plt.plot(spike_times[i])
    
plt.gca().invert_yaxis()
tick_locs = np.arange(0, len(x_labels), 10)
plt.xticks(tick_locs, x_labels[::10])
plt.show()"""