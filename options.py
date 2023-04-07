import torch


class Options(object):
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_steps = 300
        self.learning_rate = 0.001
        self.hidden_size = [20, 10, 10, 10, 10, 10]
        self.batch_size = 64
        self.num_epochs = 50