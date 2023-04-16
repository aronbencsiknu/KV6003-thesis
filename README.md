# Detecting Stock Market Manipulation using Deep Spiking Neural Networks

This is the practical work for my undergraduate thesis at Northumbria University, Newcastle.

## Installation

1. Install [Python](https://www.python.org/downloads/)
2. Install the required libraries using the command ```pip install -r requirements.txt```

## Usage
### Flags
main.py can be run using the following flags:
- ```--output_decoding``` Method of interpreting output spike-trains.
- ```--input_encoding``` Method of encoding the input data into spike-trains.
- ```--wandb_logging``` Enable logging to [Weights&Biases](https://www.wandb.ai)
  - ```--wandb_project``` Weights&Biases project name
  - ```--wandb_entity```  Weights&Biases entity name (username)
- ```--net_type``` Type of neural network (SNN, CSNN, CNN, OC_CSNN)
- ```--num_epochs``` Number of training epochs
- ```--batch_size``` Batch size
- ```--num_steps``` Number of time-steps for spike encoding. Overwritten for population encoding
- ```--learning_rate``` Learning rate of the optimizer
- ```--hidden_size``` Hidden size of the defined network
- ```--neuron_type``` Type of spiking neuron model (Leaky, Synaptic)
- ```--window_length``` Lenght of window to slice the input data stream
- ```--window_overlap``` Overlap of the sliding window
- ```--manipulation_length``` Lenght of the injected artificial manipulations
