import torch
import argparse


class Options(object):
    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.initialized = False

    def add_args(self):
        self.argparser.add_argument("--path", type=str, default="Amazon", choices=["Amazon", "Intel", "Apple", "Microsoft", "Google"], help="Dataset to use (Amazon, Intel, Apple, Microsoft, Google)")
        self.argparser.add_argument("--output_decoding", type=str, default="rate", choices=["rate", "latency"], help="input encoding (rate, latency)")
        self.argparser.add_argument("--input_encoding", type=str, default="rate", choices=["rate", "latency", "population"], help="output decoding (rate, latency, population)")
        self.argparser.add_argument("-wb", "--wandb_logging", action='store_true', help="enable logging to Weights&Biases")
        self.argparser.add_argument("--wandb_project", type=str, default="thesis_results_final", help="Weights&Biases project name")
        self.argparser.add_argument("--wandb_entity", type=str, default="aronbencsik", help="Weights&Biases entity name")
        self.argparser.add_argument("--load_name", type=str, default="final", help="Name of the model, hyperparameter dictionary and gain values to load")
        self.argparser.add_argument("--net_type", type=str, default="SNN",choices=["SNN", "CSNN", "CNN", "OC_SCNN", "RNN"], help="Type of network to use (SNN, CSNN, CNN, OC_SCNN, RNN)")
        self.argparser.add_argument("-sm", "--save_model", action='store_true', help="Save the model when finished training")
        self.argparser.add_argument("-lm", "--load_model", action='store_true', help="Skip training and load a model")
        self.argparser.add_argument("-ld", "--load_model_dict", action='store_true', help="Load a model hyperparameter dictionary")
        self.argparser.add_argument("-r", "--real_time", action='store_true', help="Skip training and load a model")
        self.argparser.add_argument("-sw", "--sweep", action='store_true', help="Perform a hyperparameter sweep with Weights&Biases")
        self.argparser.add_argument("-sr", "--save_results", action='store_true', help="Perform a hyperparameter sweep with Weights&Biases")
        self.argparser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
        self.argparser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        self.argparser.add_argument("--num_steps", type=int, default=100, help="Number of time steps to simulate")
        self.argparser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
        self.argparser.add_argument("--hidden_size", nargs='+',default=[128, 128], help='Hidden layer size')
        self.argparser.add_argument("--neuron_type", type=str, default="Leaky", choices=["Leaky","Synaptic"], help="Type of neuron to use (Leaky, Synaptic)")
        self.argparser.add_argument("--window_length", type=int, default=10, help="Length of the window to use for the dataset")
        self.argparser.add_argument("--window_overlap", type=float, default=0.0, help="Overlap of windows in the dataset")
        self.argparser.add_argument("--train_method", type=str, default="multiclass", choices=["multiclass", "oneclass"], help="Method to use for training (multiclass, oneclass)")
        self.argparser.add_argument("--manipulation_length", type=int, default=3, help="Length of the injected manipulations")
        self.argparser.add_argument("--subset_indeces", nargs='+',default=[4,6], help='Select which features to use')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.add_args()
        self.opt = self.argparser.parse_args()

        self.opt.hidden_size = [int(i) for i in self.opt.hidden_size]
        self.opt.subset_indeces = [int(i) for i in self.opt.subset_indeces]

        self.opt.wandb_key = "edfb94e4b9dca47c397a343d2829e9af262d9e32"

        # conditionally set option values
        if self.opt.net_type == "CSNN" or self.opt.net_type == "OC_SCNN":
            self.opt.flatten_data = False
            self.opt.set_type = "spiking"

        elif self.opt.net_type == "SNN":

            if self.opt.input_encoding != "population":
                self.opt.flatten_data = True
                self.opt.set_type = "spiking"

            else:
                self.opt.flatten_data = False
                self.opt.set_type = "spiking"

        elif self.opt.net_type == "CNN" or self.opt.net_type == "RNN":
            self.opt.flatten_data = False
            self.opt.set_type = "non-spiking"

        # Set the run name for saving models, plots, etc.
        self.opt.run_name = self.opt.net_type + "_" + self.opt.input_encoding + "-to-" + self.opt.output_decoding + "_" + self.opt.neuron_type + "_price30-volume4"
        self.opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        return self.opt