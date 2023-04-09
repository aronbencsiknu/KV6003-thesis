import torch
from snntorch import functional


class Options(object):
    def __init__(self, parsed_args):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_steps = parsed_args.num_steps
        self.learning_rate = parsed_args.learning_rate
        self.hidden_size = parsed_args.hidden_size
        self.batch_size = parsed_args.batch_size
        self.num_epochs = parsed_args.num_epochs
        self.input_encoding = parsed_args.input_encoding
        self.output_decoding = parsed_args.output_decoding
        self.net_type = parsed_args.net_type
        self.wandb_logging = parsed_args.wandb_logging
        self.wandb_project = parsed_args.wandb_project
        self.wandb_entity = parsed_args.wandb_entity
        self.neuron_type = parsed_args.neuron_type
        self.window_length = parsed_args.window_length

        self.wandb_key = "edfb94e4b9dca47c397a343d2829e9af262d9e32"

        if self.net_type == "CSNN":
            self.flatten_data = False
            self.set_type = "spiking"

        elif self.net_type == "SNN":
            self.flatten_data = True
            self.set_type = "spiking"

        elif self.net_type == "CNN":
            self.flatten_data = False
            self.set_type = "non-spiking"

        self.wandb_group = self.net_type + "_" + self.input_encoding + "_" + self.output_decoding + "_" + self.neuron_type