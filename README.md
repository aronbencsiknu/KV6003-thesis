# Detecting Stock Market Manipulation Using Biologically Plausible Deep Neural Networks

The practical work for my undergraduate thesis (KV6003) at Northumbria University, Newcastle.

[Thesis Report](https://drive.google.com/file/d/1ZEcvkLQGcEgICU87VQKgOC2bJsX2Xv68/view?usp=sharing) 

## Setup

Create conda environment:
```
conda env create -f environment.yml
conda activate snn_aronbencsik
```

Alternatively, the required libraries can be installed via pip:
```
pip3 install -r requirements.txt
```

## Examples

### Load a pre-trained model and run testing
With temporal population encoding. Include the ```-pe``` flag to display the neuronal tuning curves.
```
python main.py --input_encoding population -lm
```
With rate coding.
```
python main.py --input_encoding rate -lm
```
And latency encoding.
```
python main.py --input_encoding latency -lm
```
### Train a new model
Train a new feedforward SNN model with the temporal population encoding. Save the classification metrics and the trained model.
```
python main.py --input_encoding population --num_epochs 500 -ld -sr -sm
```

## Usage

The program can be run in 6 main ways:

| Argument  | Description |
| ------------- | ------------- |
| ```python main.py``` | Without specifying any command line arguments trains a model with default arguments |
| ```python main.py -sm``` | Run a training and save the model with the lowest loss |
| ```python main.py -lm``` | Load a pre-trained model and skips training |
| ```python main.py -ld``` | Load a hyperparameter dictionary and trains a model initialized with the specified hyperparams |
| ```python main.py -rt``` | Run an inference with neuron states persisted over windows |
| ```python main.py -sw``` | Run a hyperparameter sweep |

Additionally, the following arguments can be specified to customize the training or testing:

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
| ```--plot_tuning_curves``` | Plot neuronal tuning curves for ensembles |
