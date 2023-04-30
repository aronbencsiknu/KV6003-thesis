# Detecting Stock Market Manipulation using Deep Spiking Neural Networks

This is the practical work for my undergraduate thesis at Northumbria University, Newcastle.

## Installation

1. Install [Python](https://www.python.org/downloads/)
2. Install the required libraries using the command ```pip install -r requirements.txt```

## Usage
The program can be run in 3 main ways:
1. Running ```python main.py``` without specifying any command line arguments trains a model with default arguments.
2. Running ```python main.py -sm``` saves the trained model.
3 ```python main.py -lm``` loads a pre-trained model and skips training
4. ```python main.py -ld``` loads a hyperparameter dictionary and trains a model initialized with the specified hyperparams. If population encoding is selected, the gains are also loaded.
5. ```python main.py -rt```
6. ```python main.py -sw``` runs a hyperparameter sweep

Additionally, the following command line arguments can be used to customize the run:
- ```--output_decoding``` Method of interpreting output spike-trains.
- ```--input_encoding``` Method of encoding the input data into spike-trains.
- ```--wandb_logging``` Enable logging to [Weights&Biases](https://www.wandb.ai)
  - ```--wandb_project``` Weights&Biases project name
  - ```--wandb_entity```  Weights&Biases entity name (username)
- ```--net_type``` Type of neural network
- ```--num_epochs``` Number of training epochs
- ```--batch_size``` Batch size
- ```--learning_rate``` Learning rate of the optimizer
- ```--hidden_size``` Hidden size of the defined network
- ```--neuron_type``` Type of spiking neuron model (Leaky, Synaptic)
- ```--window_length``` Lenght of window to slice the input data stream
- ```--window_overlap``` Overlap of the sliding window
- ```--manipulation_length``` Lenght of the injected artificial manipulations
- ```--subset_indeces``` Select which features to use
