import torch.nn as nn
import torch
import snntorch as snn
from snntorch import surrogate
import numpy as np


"""
##############################################
Feedforward SNN with leaky or synaptic neurons
##############################################
"""
class SNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, beta=0.5, alpha=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Synaptic"):
        super().__init__()

        self.neuron_type = neuron_type
        self.synapses = nn.ModuleList() # list of synapses
        self.neurons = nn.ModuleList() # list of neurons

        hidden_size.insert(0, input_size)
        hidden_size.append(output_size)

        size = hidden_size
        
        for i in range(len(size) - 1):
            self.synapses.append(nn.Linear(size[i], size[i + 1], bias=False))
            if self.neuron_type == "Leaky":
                self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad))

            elif self.neuron_type == "Synaptic":
                self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.2, spike_grad=spike_grad, learn_threshold=True))
        
        self.dropout = nn.Dropout(p=0.4)

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
                if i != 0 and i != len(self.neurons) - 1:
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


"""
################################################
Convolutional SNN with leaky or synaptic neurons
################################################
"""
class CSNN(nn.Module):
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

        self.synapses.append(nn.Linear(2*hidden_size[-1], 256))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))

        self.synapses.append(nn.Linear(256, 2))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))
        self.lsm = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x, num_steps, time_first=False, input_gaussian=False):

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

            if not input_gaussian:
                for i in range(len(self.hidden_size) - 1):
                    
                    current = self.synapses[i](spikes) # pass spikes through synapses
                    current = self.pools[i](current) # pooling currents

                    if self.neuron_type == "Leaky":
                        spikes, membranes[i] = self.neurons[i](current, membranes[i])
                    
                    elif self.neuron_type == "Synaptic":
                        spikes, syns[i], membranes[i] = self.neurons[i](current, syns[i], membranes[i])

                    # record output layer membrane potentials and spikes
                    mem_recs[i].append(torch.flatten(membranes[i].clone(), start_dim=1))
                    spk_recs[i].append(torch.flatten(spikes.clone(), start_dim=1))

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
    
    def generate_gaussian_feature(self):
        gaussian_dist = np.random.normal(size=(self.batch_size, 2 * self.hidden_size[-1]))
        max = np.asarray(gaussian_dist[i]).max()
        min = np.asarray(gaussian_dist[i]).min()

        for i in range(gaussian_dist[i]):
            # normalizing values as: value' = (value - min) / (max - min)
            gaussian_dist[i] = (gaussian_dist[i] - min) / (max - min)

        return gaussian_dist

"""
###############################
Conventional CNN for comparison
###############################
"""
class CNN(nn.Module):
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
    

"""
###############################
GAUSSIAN TEST
###############################
"""
class FeatureExtractorBody(nn.Module):
    def __init__(self, batch_size, hidden_size=[4,32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        super(FeatureExtractorBody, self).__init__()

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

    def init_neurons(self):
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

        return [membranes, syns, mem_recs, spk_recs, syn_recs]

    def forward(self, x, membranes, syns, mem_recs, spk_recs):
        spikes = x # input spikes

        for i in range(len(self.hidden_size) - 1):
            
            current = self.synapses[i](spikes) # pass spikes through synapses
            current = self.pools[i](current) # pooling currents

            if self.neuron_type == "Leaky":
                spikes, membranes[i] = self.neurons[i](current, membranes[i])
            
            elif self.neuron_type == "Synaptic":
                spikes, syns[i], membranes[i] = self.neurons[i](current, syns[i], membranes[i])

            # record output layer membrane potentials and spikes
            mem_recs[i].append(torch.flatten(membranes[i].clone(), start_dim=1))
            spk_recs[i].append(torch.flatten(spikes.clone(), start_dim=1))

        return [spikes, spk_recs, mem_recs]
    
class ClassificationHead(nn.Module):
    def __init__(self, batch_size, synapses, neurons, pools, hidden_size=[4,32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        super(ClassificationHead, self).__init__()

        self.batch_size = batch_size
        self.neuron_type = neuron_type
        self.hidden_size = hidden_size

        self.neurons = neurons # list of neurons
        self.pools = pools # list of max pooling layers
        self.synapses = synapses # list of synapses

        self.adaptive_pool = nn.AdaptiveAvgPool1d(2)

        self.synapses.append(nn.Linear(2*hidden_size[-1], 256))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))

        self.synapses.append(nn.Linear(256, 2))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))
        self.lsm = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, spikes, membranes, syns, mem_recs, spk_recs):

        spikes = self.adaptive_pool(spikes)
        spikes = torch.flatten(spikes, start_dim=1)

        for i in range(len(self.hidden_size) - 1 , len(self.neurons)):

            current = self.synapses[i](spikes)

            if self.neuron_type == "Leaky":
                spikes, membranes[i] = self.neurons[i](current, membranes[i])
            
            elif self.neuron_type == "Synaptic":
                spikes, syns[i], membranes[i] = self.neurons[i](current, syns[i], membranes[i])

            mem_recs[i].append(membranes[i].clone())
            spk_recs[i].append(spikes.clone())

        return [spk_recs, mem_recs]
    
    def init_neurons(self):
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

        return [membranes, syns, mem_recs, spk_recs, syn_recs]

class CSNNGaussian(nn.Module):
    def __init__(self, batch_size, hidden_size=[4,32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        super(CSNNGaussian, self).__init__()

        self.feature_extractor_body = FeatureExtractorBody(batch_size, hidden_size, beta, spike_grad, neuron_type)
        self.classification_head = ClassificationHead(batch_size, self.feature_extractor_body.synapses, self.feature_extractor_body.neurons, self.feature_extractor_body.pools, hidden_size, beta, spike_grad, neuron_type)

    def forward(self, x, num_steps, time_first=False):

        if not time_first:
          x=x.transpose(1, 0) # convert to time first, batch second
        
        membranes, sny, mem_recs, spk_recs, syn_recs = self.feature_extractor_body.init_neurons()
        membranes, syns, mem_recs, spk_recs, syn_recs = self.classification_head.init_neurons()

        for step in range(num_steps):
            spikes = x[step]
            spikes, spk_recs, mem_recs = self.feature_extractor_body(spikes, membranes, syns, mem_recs, spk_recs)
            spk_recs, mem_recs = self.classification_head(spikes, membranes, syns, mem_recs, spk_recs)

        for i in range(len(spk_recs)):
            mem_recs[i] = torch.stack(mem_recs[i], dim=0)
            spk_recs[i] = torch.stack(spk_recs[i], dim=0)

        return [spk_recs, mem_recs]