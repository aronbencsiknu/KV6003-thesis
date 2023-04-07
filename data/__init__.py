import csv
import pathlib
import numpy as np

def create_dataset():
    dataset = LobsterData()
    return dataset


class LobsterData:
    def __init__(self, path=None):

        #self.path = path
        self.path = pathlib.Path(pathlib.Path.cwd() / "data" / "LOBSTER_SampleFile_AMZN_2012-06-21_1/")

        self.orderbook_path = self.path / "AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
        self.message_path = self.path / "AMZN_2012-06-21_34200000_57600000_message_1.csv"

        self.orderbook_data = []
        self.message_data = []

        with open(self.message_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.message_data.append(row)

        with open(self.orderbook_path, 'r') as file:
            reader = csv.reader(file)
            limiter = 0
            for row in reader:
                if limiter < 10000:
                    self.orderbook_data.append(row)
                #self.orderbook_data.append(row)
                
                limiter += 1

        self.orderbook_data = np.array(np.transpose(self.orderbook_data), dtype=np.float32)

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.orderbook_data)
