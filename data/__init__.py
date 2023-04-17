import csv
import pathlib
import numpy as np

class LobsterData:
    def __init__(self, path, limit=None):
        
        self.path = pathlib.Path(pathlib.Path.cwd() / "data" / path)

        self.orderbook_path = self.path / "orderbook.csv"
        self.message_path = self.path / "message.csv"

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
                if limiter < 20000:
                    self.orderbook_data.append(row)
                
                limiter += 1

        self.orderbook_data = np.array(np.transpose(self.orderbook_data), dtype=np.float32)

    def __len__(self):

        """Return the number of data in the dataset"""
        return len(self.orderbook_data)
