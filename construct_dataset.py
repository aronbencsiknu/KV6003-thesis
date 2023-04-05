import matplotlib.pyplot as plt
import numpy as np
import copy
import pywt
import torch
from torch.utils.data import DataLoader, Dataset
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import functional
from snntorch import surrogate
from snntorch import backprop


class ManipulatedDataset:
    def __init__(self, original_data):

        # manipulation characteristics
        self.m_len = 5
        self.price_increase = 1.08
        self.volume_increase = 20
        self.epsilon = 0.001
        self.random_num = 0.2

        

        self.original_data = original_data
        self.data = copy.deepcopy(original_data)
        
        self.manipulated_ask_price = self.data[0]
        self.manipulated_ask_volume = self.data[1]
        self.manipulated_bid_price = self.data[2]
        self.manipulated_bid_volume = self.data[3]

        self.manipulation_indeces = []

        for i in range(np.shape(self.data)[1]):
            if np.random.random() < self.epsilon:

                self.manipulation_indeces.append(i)

                # get initial price and volume points at var:i
                ask_P0 = self.manipulated_ask_price[i]
                ask_V0 = self.manipulated_ask_volume[i]
                bid_P0 = self.manipulated_bid_price[i]
                bid_V0 = self.manipulated_bid_volume[i]

                # generate manipulated orderbook data instance
                manipulated_orderbook = self.generate_manipulated_instance(bid_P0, ask_P0, bid_V0, ask_V0, self.m_len)
                
                # inject manipulation at var:i
                self.manipulated_ask_price = np.concatenate((self.manipulated_ask_price[:i], manipulated_orderbook[0], self.manipulated_ask_price[i:]))
                self.manipulated_ask_volume = np.concatenate((self.manipulated_ask_volume[:i], manipulated_orderbook[1], self.manipulated_ask_volume[i:]))
                self.manipulated_bid_price = np.concatenate((self.manipulated_bid_price[:i], manipulated_orderbook[2], self.manipulated_bid_price[i:]))
                self.manipulated_bid_volume = np.concatenate((self.manipulated_bid_volume[:i], manipulated_orderbook[3], self.manipulated_bid_volume[i:]))

                i+=self.m_len
            
            self.data = [self.manipulated_ask_price,
                         self.manipulated_ask_volume,
                         self.manipulated_bid_price,
                         self.manipulated_bid_volume]

    def generate_manipulated_bid_ask_price(self, P0, m_len):

        pump_len = int(m_len/3)
        dump_len = m_len - pump_len
        pumping_array = np.linspace(P0, P0 * self.price_increase, pump_len)
        dumping_array = np.linspace(P0 * self.price_increase, P0, dump_len)

        return np.concatenate((pumping_array, dumping_array))


    def generate_manipulated_bid_ask_volume(self,V0, m_len):
        pump_len = int(m_len/3)
        dump_len = m_len - pump_len

        pumping_array = np.linspace(V0, V0 * self.volume_increase, pump_len)
        dumping_array = np.linspace(V0 * self.volume_increase, V0, dump_len)

        return np.concatenate((pumping_array, dumping_array))


    def generate_manipulated_instance(self, bid_P0, ask_P0, bid_V0, ask_V0, m_len,):

        manipulated_bid_price = self.generate_manipulated_bid_ask_price(bid_P0, m_len)
        manipulated_ask_price = self.generate_manipulated_bid_ask_price(ask_P0, m_len)

        manipulated_bid_volume = self.generate_manipulated_bid_ask_volume(bid_V0, m_len)
        manipulated_ask_volume = self.generate_manipulated_bid_ask_volume(ask_V0, m_len)

        return [manipulated_bid_price, manipulated_bid_volume, manipulated_ask_price, manipulated_ask_volume]


    """def plot(ypoints, data_index):
        fig, axs = plt.subplots(nrows=len(ypoints), ncols=1, figsize=(8, len(ypoints) * 3))
        if len(ypoints) > 1:
            for i in range(len(ypoints)):
                if i % 2 == 0:
                    axs[i].plot(ypoints[i], 'g-')
                else:
                    axs[i].plot(ypoints[i], 'r-')
        else:
            axs.plot(ypoints[0])

        plt.show()"""

class ExtractFeatures:
    def __init__(self, data):
        print("\nExtracting features...")
        self.original_data = data
        self.original_bid_price = data[0]
        self.original_ask_price = data[1]
        self.original_bid_volume = data[1]
        self.original_ask_volume = data[1]

        # P_t and V_t
        self.bid_P = self.original_bid_price
        self.ask_P = self.original_ask_price
        
        self.bid_V = self.original_bid_volume
        self.ask_V = self.original_ask_volume

        # dPt/d_t and dV_t/d_t
        self.bid_P_der = self.take_derivative(self.original_bid_price)
        self.ask_P_der = self.take_derivative(self.original_ask_price)

        self.bid_V_der = self.take_derivative(self.original_bid_volume)
        self.ask_V_der = self.take_derivative(self.original_ask_volume)

        # dPhat_t/d_t and dVhat_t/d_t
        self.bid_P_der_hf = self.take_derivative(self.extract_high_frequencies(self.original_bid_price))
        self.ask_P_der_hf = self.take_derivative(self.extract_high_frequencies(self.original_ask_price))

        self.bid_V_der_hf = self.take_derivative(self.extract_high_frequencies(self.original_bid_volume))
        self.ask_V_der_hf = self.take_derivative(self.extract_high_frequencies(self.original_ask_volume))

        # Phat and Vhat
        self.bid_P_hf = self.extract_high_frequencies(self.original_bid_price)
        self.ask_P_hf = self.extract_high_frequencies(self.original_ask_price)

        self.bid_V_hf = self.extract_high_frequencies(self.original_bid_volume)
        self.ask_V_hf = self.extract_high_frequencies(self.original_ask_volume)

        self.features = [self.bid_P, self.ask_P, self.bid_V, self.ask_V, self.bid_P_der, self.ask_P_der, self.bid_V_der, self.ask_V_der, self.bid_P_der_hf, self.ask_P_der_hf, self.bid_V_der_hf, self.ask_V_der_hf, self.bid_P_hf, self.ask_P_hf, self.bid_V_hf, self.ask_V_hf]


    """def slice_data_to_windows(self, data, window_size):
        windows = []
        for i in range(0,len(data), window_size):
            chunk = data[i:i+window_size]
            windows.append(chunk)

        return windows"""


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
            gradients.append((data[i-1] - data[i+1])/2)

        gradients.append(gradients[-1])
        return gradients

class LabelledWindows:
    def __init__(self, data, manipulation_indices, window_size):
        print("\nLabelling windows...")
        self.manipulation_indices = manipulation_indices
        self.windows = self.slice_data_to_windows(data, window_size)
        self.labels = []

        for i in range(np.shape(self.windows)[0]):
            if i in self.manipulation_indices:
                self.labels.append(1)
            else:
                self.labels.append(0)

    def slice_data_to_windows(self, data, window_size):
        windows = []
        for i in range(0, np.shape(data)[1], window_size):
            window = []
            for j in range(0, np.shape(data)[0]):
                #print("j: ",j)
                chunk = data[j][i : i+window_size]
                window.append(chunk)
                #if len(chunk) == window_size:
                #    window.append(chunk)
            if np.shape(window)[1] == window_size:
                windows.append(window)

        windows = np.transpose(windows, (0,2,1))
        return windows
    
class SpikingDataset(Dataset):
    def __init__(self, data, targets, num_steps, encoding="rate"):
      print("\nCreating SpikingDataset...")
      self.data=data.copy()
      self.targets=targets.copy()
      self.encoding=encoding
      self.db=[]

      self.data = torch.FloatTensor(self.data)
      self.targets = torch.LongTensor(self.targets)

      for i in range(len(self.data)):
        item=torch.flatten(self.data[i])
        if self.encoding=="rate":
          item = spikegen.rate(item,num_steps=num_steps)
        elif self.encoding=="latency":
          item = spikegen.latency(item,num_steps=num_steps,normalize=True, linear=True)
        else:
          raise Exception("Only rate and latency encodings allowed") 

        target=self.targets[i]
        self.db.append([item,target])

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx][0]
        label = self.db[idx][1]
        return data, label