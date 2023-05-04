# Detecting Stock Market Manipulation Using Biologically Plausible Deep Neural Networks

## Summary

The practical work for my undergraduate thesis (KV6003) at Northumbria University, Newcastle.
<details>
  <summary>Read more</summary>

  The behaviour of Spiking Neural Networks (SNNs) is inspired by biological brains, where information is represented by momentary pulses of energy. This contrasts with conventional Artificial Neural Networks (ANNs), which represent data as vectors, and model the neuron soma behaviour using a non-linear activation function. The human brain achieves its remarkable computational capabilities using approximately the same amount of electricity as a compact fluorescent lightbulb. On neuromorphic hardware, SNNs also promise to be significantly more energy efficient than current ANNs, while also being particularly suited for signal processing applications. In this work, SNNs were applied to the detection of trade-based manipulation patterns in stock market data. During trade-based stock market manipulation, malicious actors use legitimate trades with an intent to influence the price of a stock for personal gains. This form of manipulation is illegal. However, since it is being conducted through legitimate instruments, detecting it is difficult. Several spiking and non-spiking architectures are implemented and their performances are evaluated on this task. Furthermore, multiple spike encoding techniques were tested, including a temporal population encoding method, which fully leverages the complex spatiotemporal properties of biologically inspired neurons. The findings suggest that biologically plausible neural networks are able to achieve state-of-the-art performance in this task. Moreover, it is shown that SNNs are capable of effectively learning temporally encoded spike trains, along with commonly used rate-coded ones.

</details>

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
With temporal population encoding:
```
python main.py --input_encoding population -lm -pe
```
Rate encoding:
```
python main.py --input_encoding rate -lm
```
And latency encoding:
```
python main.py --input_encoding latency -lm
```
### Train a new model
Train a new feedforward SNN model with the temporal population encoding. Save the classification metrics and the trained model.
```
python main.py --input_encoding population --num_epochs 500 -ld -sr -sm
```

### Run a real-time inference
Run a real-time inference with the membrane potential persisted over windows.
```
python main.py --input_encoding population -lm -rt
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
