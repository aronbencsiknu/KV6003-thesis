from data import create_dataset
import matplotlib.pyplot as plt
import numpy as np
import copy
import pywt


class ManipulatedDataset:
    def __init__(self, data):

        # manipulation characteristics
        self.m_len = 5
        self.price_increase = 1.08
        self.volume_increase = 20
        self.epsilon = 0.1
        self.random_num = 0.2

        self.original_data = data
        self.data = copy.deepcopy(data)

        for i in range(len(data)):
            if self.random_num < self.epsilon:

                # get initial price and volume points at var:i
                ask_P0 = self.original_data.orderbook_data[i][0]
                ask_V0 = self.original_data.orderbook_data[i][1]
                bid_P0 = self.original_data.orderbook_data[i][2]
                bid_V0 = self.original_data.orderbook_data[i][3]

                # generate manipulated orderbook data instance
                m_o = self.generate_manipulated_instance(bid_P0, ask_P0, bid_V0, ask_V0, self.m_len, self.price_increase, self.volume_increase)
                # inject manipulation at var:i
                self.data.orderbook_data = self.original_data.orderbook_data[:i] + m_o + self.original_data.orderbook_data[i:]

                i+=self.m_len

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


    def generate_manipulated_instance(self, bid_P0, ask_P0, bid_V0, ask_V0, m_len, price_increase, volume_increase):

        final_list = []

        manipulated_bid_price = self.generate_manipulated_bid_ask_price(bid_P0, m_len, price_increase)
        manipulated_ask_price = self.generate_manipulated_bid_ask_price(ask_P0, m_len, price_increase)

        manipulated_bid_volume = self.generate_manipulated_bid_ask_volume(bid_V0, m_len, volume_increase)
        manipulated_ask_volume = self.generate_manipulated_bid_ask_volume(ask_V0, m_len, volume_increase)

        for i in range(len(manipulated_bid_price)):
            final_list.append([manipulated_ask_price[i], manipulated_ask_volume[i], manipulated_bid_price[i], manipulated_bid_volume[i]])

        return final_list


    def plot(ypoints, data_index):
        fig, axs = plt.subplots(nrows=len(ypoints), ncols=1, figsize=(8, len(ypoints) * 3))
        if len(ypoints) > 1:
            for i in range(len(ypoints)):
                if i % 2 == 0:
                    axs[i].plot(ypoints[i], 'g-')
                else:
                    axs[i].plot(ypoints[i], 'r-')
        else:
            axs.plot(ypoints[0])

        plt.show()

class ExtractFeatures:
    def __int__(self, data):
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


    def slice_data_to_windows(self, data, window_size):
        windows = []
        for i in range(0,len(data), window_size):
            chunk = data[i:i+window_size]
            windows.append(chunk)

        return windows

    def extract_high_frequencies(self, data):
        # Apply DWT transform to the time-series data
        cA, cD = pywt.dwt(data, 'db2')

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

        return data

    def take_derivative(self, data):

        gradients = []

        for i in range(data):
            gradients.append((data[i-1] - data[i+1])/2)

        return gradients

class WindowedData:
    