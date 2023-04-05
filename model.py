import torch.nn as nn
import torch
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import functional
from snntorch import surrogate
from snntorch import backprop
import numpy as np


# Define Network

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25)):
        super().__init__()

        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()

        self.synapses.append(nn.Linear(input_size, hidden_size[0], bias=False))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.1, learn_threshold=True))

        for i in range(len(hidden_size) - 1):
            self.synapses.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=False))
            self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.1, learn_threshold=True))

        self.synapses.append(nn.Linear(hidden_size[-1], output_size, bias=False))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.1, learn_threshold=True))

        self.dropout = nn.Dropout(p=0.5)
        self.num_hidden = len(hidden_size)

    def forward(self, x, num_steps, time_first=False, plot=False):

        if not time_first:
          x=x.transpose(1, 0)

        membranes = [] # list of membrane potentials

        mem_rec = [] # list of membrane potentials over steps at output layer
        spk_rec = [] # list of spikes over steps at output layer

        for i in range(len(self.neurons)):
            membranes.append(self.neurons[i].init_leaky())
        for step in range(num_steps):
            spk = x[step] # input spikes at t=step

            for i in range(len(self.neurons)):
                spk = self.synapses[i](spk) # synapse

                if i != 0 or i != len(self.neurons) - 1:
                    spk = self.dropout(spk) # apply dropout to hidden layers

                spk, membranes[i] = self.neurons[i](spk, membranes[i]) # neuron

                if i == len(self.neurons) - 1:
                    mem_rec.append(membranes[i].clone())
                    spk_rec.append(spk.clone())
            
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    

class SNN_original(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.5):
        super().__init__()

        # Initialize layers
        self.spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(input_size, 20, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=self.spike_grad, threshold=0.1, learn_threshold=True)
        self.fc2 = nn.Linear(20, 20, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=self.spike_grad, threshold=0.1, learn_threshold=True)
        self.fc3 = nn.Linear(20, output_size, bias=False)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=self.spike_grad, threshold=0.1, learn_threshold=True)

        self.dropout = nn.Dropout(p=0.5)
        

        #self.batch_norm = snn.BatchNorm1d()

    def forward(self, x, num_steps, time_first=True, plot=False):

        if not time_first:
          x=x.transpose(1, 0)

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        mem3_rec = []

        # inference through time steps
        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)

            cur2 = self.fc2(spk1)
            cur2 = self.dropout(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)