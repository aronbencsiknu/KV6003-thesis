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
    """Feedforward SNN with leaky or synaptic neurons"""
    def __init__(self, input_size, hidden_size, output_size, beta=0.5, alpha=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Synaptic"):
        super().__init__()

        self.neuron_type = neuron_type
        self.synapses = nn.ModuleList() # list of synapses
        self.neurons = nn.ModuleList() # list of neurons

        # Input layer
        self.synapses.append(nn.Linear(input_size, hidden_size[0], bias=False))
        if self.neuron_type == "Leaky":
            self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad))
        elif self.neuron_type == "Synaptic":
            self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.2, spike_grad=spike_grad, learn_threshold=True))
        
        # Hidden layers
        for i in range(len(hidden_size) - 1):
            self.synapses.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=False))
            if self.neuron_type == "Leaky":
                self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad))
            elif self.neuron_type == "Synaptic":
                self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.2, spike_grad=spike_grad, learn_threshold=True))

        # Output layer
        self.synapses.append(nn.Linear(hidden_size[-1], output_size, bias=False))
        if self.neuron_type == "Leaky":
            self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad))
        elif self.neuron_type == "Synaptic":
            self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.2, spike_grad=spike_grad, learn_threshold=True))
        
        self.dropout = nn.Dropout(p=0.5)
        self.num_hidden = len(hidden_size)

    def forward(self, x, num_steps, time_first=False):

        if not time_first:
          x=x.transpose(1, 0) # convert to time first, batch second

        membranes = [] # initial membrane potentials
        syns = [] # initial synapse outputs

        mem_recs = [[] for _ in range(len(self.neurons))] # list of membrane potentials over steps at output layer
        spk_recs = [[] for _ in range(len(self.neurons))] # list of spikes over steps at output layer
        syn_recs = [[] for _ in range(len(self.neurons))] # list of synapse outputs over steps at output layer
        

        for i in range(len(self.neurons)):
            if self.neuron_type == "Leaky":
                membranes.append(self.neurons[i].init_leaky()) # initialize membrane potentials
            elif self.neuron_type == "Synaptic":
                temp = self.neurons[i].init_synaptic()
                syns.append(temp[0])
                membranes.append(temp[1])

        for step in range(num_steps):
            spk = x[step] # input spikes at t=step

            for i in range(len(self.neurons)):
                
                spk = self.synapses[i](spk) # pass spikes through synapses

                if i != 0 or i != len(self.neurons) - 1:
                    spk = self.dropout(spk) # apply dropout to hidden layers

                if self.neuron_type == "Leaky":
                    spk, membranes[i] = self.neurons[i](spk, membranes[i])
                
                elif self.neuron_type == "Synaptic":
                    spk, syns[i], membranes[i] = self.neurons[i](spk, syns[i], membranes[i])

                # record output layer membrane potentials and spikes
                mem_recs[i].append(membranes[i].clone())
                spk_recs[i].append(spk.clone())
            
        for i in range(len(self.neurons)):
            mem_recs[i] = torch.stack(mem_recs[i], dim=0)
            spk_recs[i] = torch.stack(spk_recs[i], dim=0)
        return [spk_recs, mem_recs]
        #return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    
class CSNN(nn.Module):
    """Convolutional Spiking Neural Network with leaky neurons"""
    def __init__(self, batch_size, hidden_size=[4,32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        super(CSNN, self).__init__()

        self.batch_size = batch_size
        self.neuron_type = neuron_type
        self.hidden_size = hidden_size

        self.neurons = nn.ModuleList() # list of neurons
        self.pools = nn.ModuleList() # list of max pooling layers
        self.synapses = nn.ModuleList() # list of synapses

        for i in range(len(hidden_size) - 1):

            self.synapses.append(nn.Conv1d(in_channels=hidden_size[i], out_channels=hidden_size[i + 1], kernel_size=3, padding=1))
            self.pools.append(nn.MaxPool1d(3, stride=2))
            self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))

        self.adaptive_pool = nn.AdaptiveAvgPool1d(2)

        self.synapses.append(nn.Linear(128, 256))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))

        self.synapses.append(nn.Linear(256, 2))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))
        self.lsm = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x, num_steps, time_first=False):

        if not time_first:
          x=x.transpose(1, 0) # convert to time first, batch second

        membranes = [] # initial membrane potentials
        syns = [] # initial synapse outputs

        mem_recs = [[] for _ in range(len(self.neurons))] # list of membrane potentials over steps at output layer
        spk_recs = [[] for _ in range(len(self.neurons))] # list of spikes over steps at output layer
        syn_recs = [[] for _ in range(len(self.neurons))] # list of synapse outputs over steps at output layer


        for i in range(len(self.neurons)):
            if self.neuron_type == "Leaky":
                membranes.append(self.neurons[i].init_leaky()) # initialize membrane potentials
            elif self.neuron_type == "Synaptic":
                temp = self.neurons[i].init_synaptic()
                syns.append(temp[0])
                membranes.append(temp[1])

        for step in range(num_steps):
            spikes = x[step] # input spikes at t=step
            for i in range(len(self.hidden_size) - 1):
                
                current = self.synapses[i](spikes) # pass spikes through synapses
                current = self.pools[i](current) # pooling currents

                if self.neuron_type == "Leaky":
                    spikes, membranes[i] = self.neurons[i](current, membranes[i])
                
                elif self.neuron_type == "Synaptic":
                    spikes, syns[i], membranes[i] = self.neurons[i](current, syns[i], membranes[i])

                # record output layer membrane potentials and spikes
                mem_recs[i].append(membranes[i].clone())
                spk_recs[i].append(spikes.clone())

            spikes = self.adaptive_pool(spikes)
            spikes = torch.flatten(spikes, start_dim=1)
            for i in range(len(self.hidden_size) - 1 , len(self.neurons)):

                current = self.synapses[i](spikes)

                if self.neuron_type == "Leaky":
                    spikes, membranes[i] = self.neurons[i](current, membranes[i])
                
                elif self.neuron_type == "Synaptic":
                    spikes, syns[i], membranes[i] = self.neurons[i](current, syns[i], membranes[i])

                # record output layer membrane potentials and spikes
                mem_recs[i].append(membranes[i].clone())
                spk_recs[i].append(spikes.clone())
            
            
        for i in range(len(self.neurons)):
            mem_recs[i] = torch.stack(mem_recs[i], dim=0)
            spk_recs[i] = torch.stack(spk_recs[i], dim=0)
            

        return [spk_recs, mem_recs]
    

class CNN(nn.Module):
    """Convenrional CNN for comparison"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        self.lsm = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)

        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)

        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.lsm(x)

        return x