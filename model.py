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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        self.lsm = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        
        #x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))

        x = self.maxpool(x)

        x = torch.relu(self.conv2(x))

        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        #x = self.maxpool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.lsm(x)
        return x