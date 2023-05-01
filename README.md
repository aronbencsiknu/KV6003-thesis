# Detecting Stock Market Manipulation Usu

This is the practical work for my undergraduate thesis at Northumbria University, Newcastle.
<details>
  <summary>Read more</summary>

  The behaviour of Spiking Neural Networks (SNNs) is inspired by biological brains, where information is represented by a series of action potentials. This contrasts with  conventional artificial neural networks (ANNs), which use vectors as inputs, and model the neuron soma behaviour using a non-linear activation function. On neuromorphic hardware, SNNs promise to be significantly more energy efficient than current ANNs, while being particularly suited for signal processing applications. During trade-based stock market manipulation, malicious actors use legitimate trades to influence a stock price for personal gains. This form of manipulation is illegal. However, since it is conducted through legitimate instruments, detecting it is difficult. In this work, SNNs are applied to the detection of trade-based manipulation patterns in stock market data. Several spiking and non-spiking architectures are implemented and their performances are compared. Furthermore, multiple spike encoding techniques are tested, including a temporal population encoding method, which fully leverages the complex spatiotemporal properties of SNNs. The findings suggest that SNNs are able to achieve state-of-the-art performance in this task. Moreover, it is shown that SNNs are capable of effectively learning temporally encoded spike trains, along with commonly used rate-coded ones.

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

## Examples

Run the feedforward SNN with the temporal population encoding. A pre-trained model is loaded, along with the optimal hyperparameter dictionary and the pre-trained encoder. The confusion matrices, spike raster plots and performance metrics are displayed.
```
python main.py --input_encoding population -ld -lm
```

Train a new feedforward SNN model with the temporal population encoding. Save the classification metrics and the trained model.
```
python main.py --input_encoding population --num_epochs 500 -ld -sr -sm
```

Run a real-time inference with the membrane potential persisted over windows.
```
python main.py --input_encoding population -ld -lm -rt
```
