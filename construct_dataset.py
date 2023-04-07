import numpy as np
import copy
import pywt
import torch
from torch.utils.data import Dataset
from snntorch import spikeplot
from snntorch import spikegen
import numpy as np


class ManipulatedDataset:
    def __init__(self, original_data):

        # manipulation characteristics
        self.m_len = 10
        self.price_increase = 30 #bps
        self.volume_increase = 5 #times
        self.epsilon = 0.03 # probability of manipulation

        self.original_data = original_data
        self.data = copy.deepcopy(original_data)
        
        self.manipulated_ask_price = self.data[0]
        self.manipulated_ask_volume = self.data[1]
        self.manipulated_bid_price = self.data[2]
        self.manipulated_bid_volume = self.data[3]

        self.manipulation_indeces = []
        i = 0
        counter = np.shape(self.data)[1]
        while counter > 1:
            i+=1
            counter-=1
            if np.random.random() < self.epsilon:

                self.manipulation_indeces.append(i)

                # get initial price and volume points at var:i
                ask_P0 = self.manipulated_ask_price[i]
                ask_V0 = self.manipulated_ask_volume[i]
                bid_P0 = self.manipulated_bid_price[i]
                bid_V0 = self.manipulated_bid_volume[i]

                # generate manipulated orderbook data instance
                manipulated_orderbook = self.generate_manipulated_instance(ask_P0, ask_V0, bid_P0, bid_V0, self.m_len)
                
                # inject manipulation at var:i
                self.manipulated_ask_price = np.concatenate((self.manipulated_ask_price[:i], manipulated_orderbook[0], self.manipulated_ask_price[i:]))
                self.manipulated_ask_volume = np.concatenate((self.manipulated_ask_volume[:i], manipulated_orderbook[1], self.manipulated_ask_volume[i:]))
                self.manipulated_bid_price = np.concatenate((self.manipulated_bid_price[:i], manipulated_orderbook[2], self.manipulated_bid_price[i:]))
                self.manipulated_bid_volume = np.concatenate((self.manipulated_bid_volume[:i], manipulated_orderbook[3], self.manipulated_bid_volume[i:]))

                i+=self.m_len
                #print(i)

        a_p = self.manipulated_ask_price
        a_v = self.manipulated_ask_volume
        b_p = self.manipulated_bid_price
        b_v = self.manipulated_bid_volume

        self.data = [a_p, a_v, b_p, b_v]

    def generate_manipulated_bid_ask_price(self, P0, m_len):

        pump_len = int(m_len/3)
        dump_len = m_len - pump_len
        pumping_array = np.linspace(P0, P0 * (1 + self.price_increase/10000), pump_len)
        dumping_array = np.linspace(P0 * (1 + self.price_increase/10000), P0, dump_len)

        return np.concatenate((pumping_array, dumping_array))


    def generate_manipulated_bid_ask_volume(self,V0, m_len):
        pump_len = int(m_len/3)
        dump_len = m_len - pump_len

        pumping_array = np.linspace(V0, V0 * self.volume_increase, pump_len)
        dumping_array = np.linspace(V0 * self.volume_increase, V0, dump_len)

        return np.concatenate((pumping_array, dumping_array))


    def generate_manipulated_instance(self, ask_P0, ask_V0, bid_P0, bid_V0, m_len):

        manipulated_bid_price = self.generate_manipulated_bid_ask_price(bid_P0, m_len)
        manipulated_ask_price = self.generate_manipulated_bid_ask_price(ask_P0, m_len)

        manipulated_bid_volume = self.generate_manipulated_bid_ask_volume(bid_V0, m_len)
        manipulated_ask_volume = self.generate_manipulated_bid_ask_volume(ask_V0, m_len)

        return [manipulated_ask_price, manipulated_ask_volume, manipulated_bid_price, manipulated_bid_volume]

class ExtractFeatures:
    def __init__(self, data):
        self.original_data = copy.deepcopy(data)
        self.original_ask_price = self.original_data[0]
        self.original_ask_volume = self.original_data[1]
        self.original_bid_price = self.original_data[2]
        self.original_bid_volume = self.original_data[3]
        self.features = []

        # P_t and V_t
        self.features.append(self.original_bid_price)
        #self.features.append(self.original_ask_price)
        self.features.append(self.original_bid_volume)
        #self.features.append(self.original_ask_volume)

        # dPt/d_t and dV_t/d_t
        self.features.append(self.take_derivative(self.original_bid_price))
        #self.features.append(self.take_derivative(self.original_ask_price))

        self.features.append(self.take_derivative(self.original_bid_volume))
        #self.features.append(self.take_derivative(self.original_ask_volume))

        """# dPhat_t/d_t and dVhat_t/d_t
        self.features.append(self.take_derivative(self.extract_high_frequencies(self.original_bid_price)))
        self.features.append(self.take_derivative(self.extract_high_frequencies(self.original_ask_price)))

        self.features.append(self.take_derivative(self.extract_high_frequencies(self.original_bid_volume)))
        self.features.append(self.take_derivative(self.extract_high_frequencies(self.original_ask_volume)))

        # Phat and Vhat
        self.features.append(self.extract_high_frequencies(self.original_bid_price))
        self.features.append(self.extract_high_frequencies(self.original_ask_price))

        self.features.append(self.extract_high_frequencies(self.original_bid_volume))
        self.features.append(self.extract_high_frequencies(self.original_ask_volume))"""

        min = len(self.features[0])
        for i in range(len(self.features)):
            if len(self.features[i]) < min:
                min = len(self.features[i])
        for i in range(len(self.features)):
            self.features[i] = self.features[i][:min]
        for i in range(len(self.features)):
            max = np.asarray(self.features[i]).max()
            min = np.asarray(self.features[i]).min()
            for j in range(len(self.features[i])):
                # normalizing values as: value' = (value - min) / (max - min)
                self.features[i][j] = (self.features[i][j] - min) / (max - min)

    def extract_high_frequencies(self, data):
        # Apply DWT transform to the time-series data
        cA, cD = pywt.dwt(data, 'db2')
        before_len = len(data)
        # Remove the low-frequency components by setting appropriate coefficients to zero
        lmbd = 0.5

        for i in range(len(cD)):
            if abs(cD[i]) > lmbd:
                cD[i] = 0

        for i in range(len(cA)):
            if abs(cA[i]) > lmbd:
                cA[i] = 0

        # Reconstruct the denoised time-series data using inverse DWT
        data = pywt.idwt(cA, cD, 'db2')
        data = data[:before_len] # remove the extra data points added by the DWT
        return data

    def take_derivative(self, data):

        gradients = []
        for i in range(len(data)-1):
            gradients.append((data[i+1] - data[i-1])/2)

        gradients.append(gradients[-1])
        return gradients

class LabelledWindows:
    def __init__(self, data, manipulation_indices, window_size):
        self.manipulation_indices = manipulation_indices
        self.windows = self.slice_data_to_windows(data, window_size)
        self.labels = []
        index = 0
        for i in range(np.shape(self.windows)[0]):
            in_range = False

            for j in range(len(manipulation_indices)):
                if index+5 < manipulation_indices[j] and manipulation_indices[j] < index+window_size-5:
                    in_range = True
                    break

            if in_range:
                self.labels.append(1)
            else:
                self.labels.append(0)

            index+=window_size

    def slice_data_to_windows(self, data, window_size):
        """
        Slices the data into windows of size window_size
        
        Parameters:
            data (np.array): The data to be sliced
            window_size (int): The size of the window"""
        
        windows = []
        for i in range(0, len(data)): # for each feature
            window = []
            for j in range(0, len(data[0]), int(window_size)): # for each window 
                if j + window_size < len(data[0]):
                    chunk = data[i][j : j+window_size]
                    window.append(chunk)
            
            windows.append(window)
            
        windows = np.transpose(np.asarray(windows), (1,2,0))
        return windows
    
class SpikingDataset(Dataset):
    def __init__(self, data, targets, num_steps, encoding="rate"):
        self.data=data.copy()
        self.targets=targets.copy()
        self.encoding=encoding
        self.db=[]
        self.n_classes = 2

        for i in range(np.shape(self.data)[0]):
            item = torch.FloatTensor(self.data[i])
            item=torch.flatten(item)

            if self.encoding=="rate":
                item = spikegen.rate(item,num_steps=num_steps)
            elif self.encoding=="latency":
                item = spikegen.latency(item,num_steps=num_steps,normalize=True, linear=True)
            else:
                raise Exception("Only rate and latency encodings allowed") 

            target=self.targets[i]
            #print("TARGET:",target)
            self.db.append([item,target])
        
        self.n_samples_per_class = self.get_class_counts()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx][0]
        label = self.db[idx][1]
        return data, label
    
    def get_class_counts(self):
        class_weights = [0] * 2
        for i in range(len(self.targets)):
            class_weights[self.targets[i]] += 1

        return class_weights
    
    def weights4balance(self):
        """
        :return: The class blanace weights for training
        """
        print("\nClass weight balancing for training.")


        w = [len(self.db) / (self.n_classes * n_curr_class) for n_curr_class in self.n_samples_per_class]
        for i, j in zip(w, [0, 1]):
            print(f"{j}\t-> {i}")

        return torch.tensor(w, dtype=torch.float32)
    
class NonSpikingDataset(Dataset):
    def __init__(self, data, targets):
        self.data=data.copy()
        self.targets=targets.copy()
        self.db=[]
        self.n_classes = 2

        for i in range(np.shape(self.data)[0]):
            item = torch.permute(torch.FloatTensor(self.data[i]),(1,0))

            target=self.targets[i]
            #print("TARGET:",target)
            self.db.append([item,target])
        
        self.n_samples_per_class = self.get_class_counts()
        print(self.n_samples_per_class)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx][0]
        label = self.db[idx][1]
        return data, label
    
    def get_class_counts(self):
        class_weights = [0] * 2
        for i in range(len(self.targets)):
            class_weights[self.targets[i]] += 1

        return class_weights
    
    def weights4balance(self):
        """
        :return: The class blanace weights for training
        """
        print("\nClass weight balancing for training.")


        w = [len(self.db) / (self.n_classes * n_curr_class) for n_curr_class in self.n_samples_per_class]
        for i, j in zip(w, [0, 1]):
            print(f"{j}\t-> {i}")

        return torch.tensor(w, dtype=torch.float32)