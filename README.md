# Detecting Stock Market Manipulation using Deep Spiking Neural Networks

This is the practical work for my undergraduate thesis at Northumbria University, Newcastle.

## Installation

Create conda environment:
```
conda env create -f environment.yml
conda activate snn_aronbencsik
```

Alternatively, the required libraries can be installed via pip:
```
pip3 install -r requirements.txt
```

## Usage
The program can be run in 6 main ways:

| Argument  | Description |
| ------------- | ------------- |
| ```python main.py``` | without specifying any command line arguments trains a model with default arguments |
| ```python main.py -sm``` | saves the trained model |
| ```python main.py -lm``` | loads a pre-trained model and skips training |
| ```python main.py -ld``` | loads a hyperparameter dictionary and trains a model initialized with the specified hyperparamsd |
| ```python main.py -rt``` |  |
| ```python main.py -sw``` | runs a hyperparameter sweep |

Additionally, the following arguments can be specified to customize the training or testing

| Argument  | Description |
| ------------- | ------------- |
| ```--output_decoding```  | Method of interpreting output spike-trains |
| ```--input_encoding```  | Method of encoding the input data into spike-trains. |
| ```--wandb_logging```  | Enable logging to [Weights&Biases](https://www.wandb.ai) |
| ```--wandb_project```  | Weights&Biases project name |
| ```--wandb_entity```  | Weights&Biases entity name (username) |
| ```--net_type```  | Type of neural network |
| ```--num_epochs``` | Number of training epochs |
| ```--batch_size``` | Batch size |
| ```--learning_rate``` | Learning rate of the optimizer |
| ```--hidden_size``` | Hidden size of the defined network |
| ```--neuron_type``` | Type of spiking neuron model (Leaky, Synaptic) |
| ```--window_length``` | Lenght of window to slice the input data stream |
| ```--window_overlap``` | Overlap of the sliding window |
| ```--manipulation_length``` | Lenght of the injected artificial manipulations |
| ```--subset_indeces``` | Select which features to use |

## Examples

```python main.py --input_encoding population -ld -lm```
