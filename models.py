import torch.nn as nn
import torch
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
import numpy as np


"""
##############################################
Feedforward SNN with leaky or synaptic neurons
##############################################
"""

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, beta=0.5, alpha=0.5, dropout=0.5, spike_grad=surrogate.atan(), neuron_type="Leaky", learn_beta=False, learn_alpha=False, learn_threshold=True, h_params=None):
        """
        :param input_size: size of input layer
        :param hidden_size: size of hidden layer
        :param output_size: size of output layer
        :param beta: membrane time constant
        :param alpha: synaptic current time constant
        :param dropout: dropout probability
        :param spike_grad: surrogate gradient
        :param neuron_type: type of neuron (Leaky or Synaptic)
        :param learn_beta: learn membrane time constant
        :param learn_alpha: learn synaptic current time constant
        :param learn_threshold: learn threshold

        :param h_params: predefined hyperparameters (overrides all other parameters)
        """

        super().__init__()

        self.synapses = nn.ModuleList() # list of synapses
        self.neurons = nn.ModuleList() # list of neurons

        # If h_params is None, use parameters passed to the constructor
        if h_params is None:
            self.neuron_type = neuron_type
            self.hidden_size = hidden_size
            
            self.learn_beta = learn_beta
            self.learn_alpha = learn_alpha
            self.learn_threshold = learn_threshold
            self.spike_grad = spike_grad
        
        # If h_params is not None, use predefined hyperparameters
        else:
            self.neuron_type = h_params["NeuronType"]
            self.hidden_size = []

            for i in range(h_params["num_hidden"]):
                self.hidden_size.append(h_params["hidden_size"])

            self.learn_beta = h_params["learn_beta"]
            self.learn_alpha = h_params["learn_alpha"]
            self.learn_threshold = h_params["learn_threshold"]

            if h_params["surrogate_gradient"] == "atan":
                self.spike_grad = surrogate.atan()

            elif h_params["surrogate_gradient"] == "sigmoid":
                self.spike_grad = surrogate.sigmoid()
            
            elif h_params["surrogate_gradient"] == "fast_sigmoid":
                self.spike_grad = surrogate.fast_sigmoid()

        self.hidden_size.insert(0, input_size)
        self.hidden_size.append(output_size)
        self.size = self.hidden_size

        for i in range(len(self.size) - 1):
            self.synapses.append(nn.Linear(self.size[i], self.size[i + 1], bias=False))
            if self.neuron_type == "Leaky":
                self.neurons.append(snn.Leaky(beta=beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, learn_threshold=self.learn_threshold))

            elif self.neuron_type == "Synaptic":
                self.neurons.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=0.1, spike_grad=self.spike_grad, learn_beta=self.learn_beta, learn_threshold=self.learn_threshold, learn_alpha=self.learn_alpha))
        
        self.dropout = nn.Dropout(p=dropout)

        self.membranes = [] # initial membrane potentials
        self.syns = [] # initial synapse outputs
        self.init_neurons()

    def init_neurons(self):
        for i in range(len(self.neurons)):
            if self.neuron_type == "Leaky":
                self.membranes.append(self.neurons[i].init_leaky()) # initialize membrane potentials
            elif self.neuron_type == "Synaptic":
                temp = self.neurons[i].init_synaptic()
                self.syns.append(temp[0])
                self.membranes.append(temp[1])

    def forward(self, x, num_steps, time_first=False, real_time=False):

        if not time_first:
          x=x.transpose(1, 0) # convert to time first, batch second

        if not real_time:
            self.membranes = [] # initial membrane potentials
            self.syns = [] # initial synapse outputs
            self.init_neurons()

        mem_recs = [[] for _ in range(len(self.neurons))] # list of membrane potentials over steps at output layer
        spk_recs = [[] for _ in range(len(self.neurons))] # list of spikes over steps at output layer
 
        for step in range(num_steps):
            spk = x[step] # input spikes at t=step

            for i in range(len(self.neurons)):
                
                spk = self.synapses[i](spk) # pass spikes through synapses
                if i != 0 and i != len(self.neurons) - 1:
                    spk = self.dropout(spk) # apply dropout to hidden layers

                if self.neuron_type == "Leaky":
                    spk, self.membranes[i] = self.neurons[i](spk, self.membranes[i])
                
                elif self.neuron_type == "Synaptic":
                    spk, self.syns[i], self.membranes[i] = self.neurons[i](spk, self.syns[i], self.membranes[i])

                # record output layer membrane potentials and spikes
                mem_recs[i].append(self.membranes[i].clone())
                spk_recs[i].append(spk.clone())
            
        for i in range(len(self.neurons)):
            mem_recs[i] = torch.stack(mem_recs[i], dim=0)
            spk_recs[i] = torch.stack(spk_recs[i], dim=0)
        return [spk_recs, mem_recs]

"""
################################################
Convolutional SNN with leaky or synaptic neurons
################################################
"""
class CSNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size=[32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        """
        :param batch_size: batch size
        :param input_size: input size
        :param hidden_size: list of hidden layer sizes
        :param beta: beta for leaky neuron
        :param spike_grad: surrogate gradient for neuron
        :param neuron_type: type of neuron (Leaky or Synaptic)
        """

        super(CSNN, self).__init__()

        self.batch_size = batch_size
        self.neuron_type = neuron_type
        self.hidden_size = hidden_size

        hidden_size.insert(0, input_size)

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

        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x, num_steps, time_first=False):

        if not time_first:
          x=x.transpose(1, 0) # convert to time first, batch second

        membranes = [] # initial membrane potentials
        syns = [] # initial synapse outputs

        mem_recs = [[] for _ in range(len(self.neurons))] # list of membrane potentials over steps at output layer
        spk_recs = [[] for _ in range(len(self.neurons))] # list of spikes over steps at output layer


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

"""
######################
One-Class SCNN attempt
######################
"""
class FeatureExtractorBody(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size=[32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        """
        Feature extractor body of the one-class SCNN

        :param batch_size: batch size
        :param input_size: input size
        :param hidden_size: list of hidden layer sizes
        :param beta: beta for leaky neuron
        :param spike_grad: surrogate gradient for neuron
        :param neuron_type: type of neuron (Leaky or Synaptic)
        """

        super(FeatureExtractorBody, self).__init__()

        hidden_size.insert(0, input_size)

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

    def forward(self, x, membranes, mem_recs, spk_recs):
        spikes = x # input spikes

        for i in range(len(self.hidden_size) - 1):
            
            current = self.synapses[i](spikes) # pass spikes through synapses
            current = self.pools[i](current) # pooling currents

            if self.neuron_type == "Leaky":
                spikes, membranes[i] = self.neurons[i](current, membranes[i])

            # record output layer membrane potentials and spikes
            mem_recs[i].append(torch.flatten(membranes[i].clone(), start_dim=1))
            spk_recs[i].append(torch.flatten(spikes.clone(), start_dim=1))

        spikes = self.adaptive_pool(spikes)
        spikes = torch.flatten(spikes, start_dim=1)

        return [spikes, membranes, spk_recs, mem_recs]
    
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

        return [membranes, mem_recs, spk_recs]


class ClassificationHead(nn.Module):
    def __init__(self, batch_size, synapses, neurons, pools, hidden_size=[32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        """
        Classification head of the one-class SCNN
        :param batch_size: batch size
        :param synapses: list of synapses
        :param neurons: list of neurons
        :param pools: list of pooling layers
        :param hidden_size: list of hidden layer sizes
        :param beta: beta for leaky neuron
        :param spike_grad: surrogate gradient for neuron
        :param neuron_type: type of neuron (Leaky or Synaptic)
        """

        super(ClassificationHead, self).__init__()

        self.batch_size = batch_size
        self.neuron_type = neuron_type
        self.hidden_size = hidden_size

        self.neurons = nn.ModuleList() # list of neurons
        self.pools = nn.ModuleList() # list of max pooling layers
        self.synapses = nn.ModuleList() # list of synapses

        self.synapses.append(nn.Linear(2*hidden_size[-1], 256))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))

        self.synapses.append(nn.Linear(256, 2))
        self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True))
        self.lsm = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(p=0.3)


    def forward(self, spikes, membranes, mem_recs, spk_recs):
        for i in range(len(self.neurons)):

            current = self.synapses[i](spikes)

            if self.neuron_type == "Leaky":
                spikes, membranes[i] = self.neurons[i](current, membranes[i])

            mem_recs[i].append(membranes[i].clone())
            spk_recs[i].append(spikes.clone())

        return [membranes, spk_recs, mem_recs]
    

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

        return [membranes, mem_recs, spk_recs]


class OC_SCNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size=[32,64], beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), neuron_type="Leaky"):
        """
        One-class SCNN parent class
        :param batch_size: batch size
        :param input_size: input size
        :param hidden_size: list of hidden layer sizes
        :param beta: beta for leaky neuron
        :param spike_grad: surrogate gradient for neuron
        :param neuron_type: type of neuron (Leaky or Synaptic)
        """

        super(OC_SCNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.neuron_type = neuron_type
        self.beta = beta
        self.spike_grad = spike_grad

        self.feature_extractor_body = FeatureExtractorBody(batch_size, input_size, hidden_size, beta, spike_grad, neuron_type)
        self.classification_head = ClassificationHead(batch_size, self.feature_extractor_body.synapses, self.feature_extractor_body.neurons, self.feature_extractor_body.pools, hidden_size, beta, spike_grad, neuron_type)

    def forward(self, x, num_steps, time_first=False, gaussian=False):

        if not time_first:
          x=x.transpose(1, 0) # convert to time first, batch second
        
        # initialize neurons
        if not gaussian:
            membranes_FE, mem_recs_FE, spk_recs_FE = self.feature_extractor_body.init_neurons()

        membranes_FC, mem_recs_FC, spk_recs_FC = self.classification_head.init_neurons()

        for step in range(num_steps):
            spikes = x[step]
            if not gaussian:
                spikes, membranes_FE, spk_recs_FE, mem_recs_FE = self.feature_extractor_body(spikes, membranes_FE, mem_recs_FE, spk_recs_FE)
            membranes_FC, spk_recs_FC, mem_recs_FC = self.classification_head(spikes, membranes_FC, mem_recs_FC, spk_recs_FC)

        for i in range(len(spk_recs_FC)):
            mem_recs_FC[i] = torch.stack(mem_recs_FC[i], dim=0)
            spk_recs_FC[i] = torch.stack(spk_recs_FC[i], dim=0)

        return [spk_recs_FC, mem_recs_FC]
    
    def freeze_body(self):
        for param in self.feature_extractor_body.parameters():
            param.requires_grad = False

    def unfreeze_body(self):
        for param in self.feature_extractor_body.parameters():
            param.requires_grad = True

    def reset_head(self):
        self.classification_head = ClassificationHead(self.batch_size, self.feature_extractor_body.synapses, self.feature_extractor_body.neurons, self.feature_extractor_body.pools, self.hidden_size, self.beta, self.spike_grad, self.neuron_type)

    def generate_gaussian_feature(self, batch_size, num_steps):

        gaussian_features = []

        gaussian_dist = np.random.normal(0, 1, 2 * self.hidden_size[-1])
        gaussian_dist = torch.FloatTensor(gaussian_dist)
        for i in range(batch_size):
            gaussian_feature = []
            for j in range(len(gaussian_dist)):
                
                item = spikegen.rate(gaussian_dist[i],num_steps=num_steps)
                gaussian_feature.append(item)

            gaussian_features.append(torch.stack(gaussian_feature, dim=0))
            
        gaussian_features = torch.stack(gaussian_features, dim=0)
        gaussian_features = torch.permute(gaussian_features, (0,2,1))

        return gaussian_features
    

"""
###############################
Conventional CNN for comparison
###############################
"""
class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
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
####################################
RNN with LSTM modules for comparison
####################################
"""
class RNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128, 2)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x.permute(1,0,2)[-1]
        x = self.linear(x)
        x = self.lsm(x)
        
        return x