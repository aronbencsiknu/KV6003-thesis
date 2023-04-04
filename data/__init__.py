import csv
import pathlib


def create_dataset():
    dataset = LobsterData()
    return dataset


class LobsterData:
    def __init__(self, path=None):

        #self.path = path
        self.path = pathlib.Path(pathlib.Path.cwd() / "data" / "LOBSTER_SampleFile_GOOG_2012-06-21_1/")

        self.orderbook_path = self.path / "GOOG_2012-06-21_34200000_57600000_orderbook_1.csv"
        self.message_path = self.path / "GOOG_2012-06-21_34200000_57600000_message_1.csv"

        self.orderbook_data = []
        self.message_data = []

        with open(self.message_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.message_data.append(row)

        with open(self.orderbook_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.orderbook_data.append(row)

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.orderbook_data)
