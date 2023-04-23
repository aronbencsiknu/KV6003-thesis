import numpy as np
import matplotlib.pyplot as plt
from snntorch import spikeplot
from torch import nn
import torch
import random
import math
from matplotlib.ticker import MaxNLocator
from progress.bar import ShadyBar

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
        
        if not self.fired:

            # Update membrane potential using Euler method
            self.V = math.exp(-dt/self.tau) * self.V + self.g * Iext
            self.voltages.append(self.V)
            
            # Generate output spike if membrane potential reaches threshold
            if self.V >= self.Vth:
                self.spike = 1
                self.fired = True
                self.time_to_fire = index
            else:
                self.spike = 0
        else:
            self.spike = 0

        return self.spike
        
    def tune_forward(self, Iext, dt, index):

        # Update membrane potential using Euler method
        self.V = math.exp(-dt/self.tau) * self.V + self.g * Iext
        self.voltages.append(self.V)
        
        # Generate output spike if membrane potential reaches threshold
        if self.V >= self.Vth:
            self.spike = 1
            self.fired = True
            self.time_to_fire = index
        else:
            self.spike = 0

        return self.spike


class CUBAPopulation():
    def __init__(self, population_size, tau=0.1, g=-5.0, Vth=1, dt=0.001, T=0.1, intercept_low=0.0, intercept_high=0.1):

        self.population_size = population_size
        self.tau = tau
        self.g = g
        self.Vth = Vth
        self.dt = dt
        self.T = T
        self.intercept_low = intercept_low
        self.intercept_high = intercept_high

        self.neurons = []

        self.gains = self.tune_receptive_fields(population_size, intercept_low, intercept_high, tau, dt, T, Vth)
        self.create_neurons()       
    
    def create_neurons(self):
        for i in range(0, self.population_size):
            self.neurons.append(CUBANeuron(tau=self.tau, g=self.gains[i], Vth=self.Vth, dt=self.dt, T=self.T))

    def tune_receptive_fields(self, count, intercept_low, intercept_high, tau, dt, T, Vth):
        gain = []
        tau_list = np.linspace(0.1, 0.9, num=count)

        if intercept_high > 0.1:
            intercepts = np.linspace(intercept_low + 1e-2, intercept_high, num=count)
        else:
            intercepts = np.logspace(np.log10(intercept_low + 1e-3), np.log10(0.2), num=count)
        
        print()
        bar = ShadyBar("Tuning neurons", max=count)
        for i in range(count):
            bar.next()
            intercept_current = intercepts[i]

            if intercept_current > 0:
                multiplier = 1
            else:
                multiplier = -1
            
            g_min = multiplier * 200

            last_voltage = Vth + 2
            counter = 0
            while last_voltage >= Vth + 1e-5:
                counter += 1
                g_min = g_min - ((last_voltage - Vth)/32) * multiplier
                test_neuron = CUBANeuron(tau_list[i], g_min, Vth, dt, T)
                for j in range(0, int(T/dt)):
                    test_neuron.tune_forward(intercept_current, dt, j)

                last_voltage = test_neuron.voltages[-1]

            gain.append(g_min)

        bar.finish()
        return gain
    
    def reset_neurons(self):
        self.neurons = []
        self.create_neurons()

    def forward(self, Iext, index):
        spikes_at_t = []
        for i in range(0, self.population_size):
            spikes_at_t.append(self.neurons[i](Iext, self.dt, index))

        return spikes_at_t
    
    def display_tuning_curve(self):
        x = np.linspace(0, 1, 500)
        y = [[] for _ in range(len(self.neurons))]

        timesteps = int(self.T/self.dt)

        for i in range(len(self.neurons)):
            for j in range(len(x)):
                for step in range(timesteps):

                    self.neurons[i].forward(x[j], self.dt, step)
                
                y[i].append(self.neurons[i].time_to_fire)
                self.reset_neurons()
        
        y = np.swapaxes(y, 0, 1)

        return x, y


class CUBALayer():
    def __init__(self, feature_dimensionality, population_size, means, tau=0.8, g=-5.0, Vth=1, dt=0.01, T=0.1):

        self.feature_dimensionality = feature_dimensionality # number of populations
        self.population_size = population_size # number of neurons in each population
        self.tau = tau
        self.g = g
        self.Vth = Vth
        self.dt = dt
        self.T = T

        self.populations = []

        for i in range(self.feature_dimensionality):
            self.populations.append(CUBAPopulation(population_size, tau, g, Vth, dt, T, intercept_low=0.0, intercept_high=means[i]))
            

    def forward(self, Iext, index=None):

        spikes_at_t = []

        # loop through features and forward the respective neural populations
        for i in range(0, self.feature_dimensionality):

            # get spikes at time t form population i from feature i in Iext 
            spikes_at_t.extend(self.populations[i].forward(Iext[i], index))

        return spikes_at_t
    
    def encode_window(self, Iext, window_size):
        Iext = np.swapaxes(Iext,1,0)
        timesteps = int(self.T/self.dt)
        spk_rec = []

        for i in range(window_size):
            for j in range(timesteps):
                temp = self.forward(Iext[i])
                spk_rec.append(temp)
            
            self.reset_neurons()

        return spk_rec

    def reset_neurons(self):
        for i in range(self.feature_dimensionality):
            self.populations[i].reset_neurons()

    def display_tuning_curves(self):
        for i in range(self.feature_dimensionality):
            plt.subplot(1, self.feature_dimensionality, i+1)
            temp = self.populations[i].display_tuning_curve()
            plt.plot(temp[0],temp[1] , linewidth=0.9)
            plt.ylabel("Time")
            plt.xlabel("Current")
        
        plt.show()

