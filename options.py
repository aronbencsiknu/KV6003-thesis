import torch
from snntorch import functional
import argparse


class Options(object):
    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.initialized = False

    def add_args(self):
        self.argparser.add_argument("--output_decoding", type=str, default="rate", choices=["rate", "latency"], help="input encoding (rate, latency)")
        self.argparser.add_argument("--input_encoding", type=str, default="rate", choices=["rate", "latency", "population"], help="output decoding (rate, latency, population)")
        self.argparser.add_argument("-wb", "--wandb_logging", action='store_true', help="enable logging to Weights&Biases")
        self.argparser.add_argument("--wandb_project", type=str, default="spiking-neural-network-experiments", help="Weights&Biases project name")
        self.argparser.add_argument("--wandb_entity", type=str, default="aronbencsik", help="Weights&Biases entity name")
        self.argparser.add_argument("--net_type", type=str, default="CSNN",choices=["SNN", "CSNN", "CNN"], help="Type of network to use (SNN, CSNN, CNN)")
        self.argparser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
        self.argparser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        self.argparser.add_argument("--num_steps", type=int, default=100, help="Number of time steps to simulate")
        self.argparser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
        self.argparser.add_argument("--hidden_size", type=int, default=[100, 100, 100], help="Hidden layer size")
        self.argparser.add_argument("--neuron_type", type=str, default="Leaky", choices=["Leaky","Synaptic"], help="Type of neuron to use (Leaky, Synaptic)")
        self.argparser.add_argument("--window_length", type=int, default=20, help="Length of the window to use for the dataset")
        self.argparser.add_argument("--window_overlap", type=float, default=0, help="Overlap of windows in the dataset")
        self.argparser.add_argument("--train_method", type=str, default="multiclass", choices=["multiclass", "oneclass"], help="Method to use for training (multiclass, oneclass)")
        self.argparser.add_argument("-i", "--inject_manipulations", action='store_true', help="Enable manipulattion injection")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.add_args()
        self.opt = self.argparser.parse_args()

        self.opt.wandb_key = "edfb94e4b9dca47c397a343d2829e9af262d9e32"

        if self.opt.net_type == "CSNN":
            self.opt.flatten_data = False
            self.opt.set_type = "spiking"

        elif self.opt.net_type == "SNN":

            if self.opt.input_encoding != "population":
                self.opt.flatten_data = True
                self.opt.set_type = "spiking"

            else:
                self.opt.flatten_data = False
                self.opt.set_type = "spiking"

        elif self.opt.net_type == "CNN":
            self.opt.flatten_data = False
            self.opt.set_type = "non-spiking"

        self.opt.wandb_group = self.opt.net_type + "_" + self.opt.input_encoding + "_" + self.opt.output_decoding + "_" + self.opt.neuron_type
        self.opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        return self.opt