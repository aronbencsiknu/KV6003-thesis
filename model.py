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
    def __init__(self, input_size, hidden_size, output_size, beta=0.5, alpha=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Synaptic"):
        super().__init__()

        self.neuron_type = neuron_type

        # Construct network
        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()

        self.synapses.append(nn.Linear(input_size, hidden_size[0], bias=False))

        if self.neuron_type == "Leaky":
            self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad))

        elif self.neuron_type == "Synaptic":
            self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.1, spike_grad=spike_grad, learn_threshold=True))
        
        for i in range(len(hidden_size) - 1):
            self.synapses.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=False))
            if self.neuron_type == "Leaky":
                self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad))

            elif self.neuron_type == "Synaptic":
                self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.1, spike_grad=spike_grad, learn_threshold=True))

        self.synapses.append(nn.Linear(hidden_size[-1], output_size, bias=False))
        if self.neuron_type == "Leaky":
            self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad))

        elif self.neuron_type == "Synaptic":
            self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.1, spike_grad=spike_grad, learn_threshold=True))

        self.dropout = nn.Dropout(p=0.2)
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
        return spk_recs, mem_recs
        #return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

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
    
class CSNN(nn.Module):
    def __init__(self, batch_size, beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25)):
        super(CSNN, self).__init__()

        self.batch_size = batch_size

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

        self.fc1 = nn.Linear(512, 256)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(256, 2)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lsm = nn.LogSoftmax(dim=1)


        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x, num_steps, time_first=False):

        if not time_first:
          x=x.transpose(1, 0) # convert to time first, batch second

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk_rec = []
        mem_rec = []
        
        for step in range(num_steps):
            spk0 = x[step] # input spikes at t=step

            x = self.conv1(spk0)
            cur1 = self.maxpool(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            x = self.conv2(spk1)
            cur2 = self.maxpool(x)
            spk2, mem2 = self.lif2(cur2, mem2)

            x = self.conv3(spk2)
            cur3 = self.adaptive_pool(x)
            spk3, mem3 = self.lif3(cur3, mem3)

            #x = torch.flatten(spk3, start_dim=1)
            spk3 = spk3.view(self.batch_size, -1)
            cur4 = self.fc1(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc1(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            spk_rec.append(spk5.clone())
            mem_rec.append(mem5.clone())
            

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)