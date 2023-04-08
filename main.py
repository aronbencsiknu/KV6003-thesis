# local imports
from data import LobsterData
from construct_dataset import ManipulatedDataset
from construct_dataset import ExtractFeatures
from construct_dataset import LabelledWindows
from construct_dataset import SpikingDataset, CustomDataset
from model import SNN, CSNN, CNN
from options import Options
from plot_spike_trains import RasterPlot, ManipulationPlot

# external imports
from snntorch import functional
import torch
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import wandb


argparser = argparse.ArgumentParser()
argparser.add_argument("--output_decoding", type=str, default="rate", help="rate or latency encoding")
argparser.add_argument("--input_encoding", type=str, default="rate", help="rate or latency decoding")
argparser.add_argument("--wandb_logging", action='store_true', help="enable logging to Weights&Biases")
argparser.add_argument("--wandb_project", type=str, default="spiking-neural-network", help="Weights&Biases project name")
argparser.add_argument("--wandb_entity", type=str, default="aronbencsik", help="Weights&Biases entity name")
argparser.add_argument("--net_type", type=str, default="CSNN", help="Type of network to use (SNN, CSNN, CNN)")
argparser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
argparser.add_argument("--batch_size", type=int, default=64, help="Batch size")
argparser.add_argument("--num_steps", type=int, default=100, help="Number of time steps to simulate")
argparser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
argparser.add_argument("--hidden_size", type=int, default=[100, 100, 100], help="Hidden layer size")
argparser.add_argument("--neuron_type", type=str, default="Leaky", help="Type of neuron to use (Leaky, Synaptic)")


parsed_args = argparser.parse_args()
opt = Options(parsed_args)

device = opt.device

unmanipulated_data = LobsterData()
manipulated_data = ManipulatedDataset(unmanipulated_data.orderbook_data)

#ManipulationPlot(manipulated_data)

extracted_features = ExtractFeatures(manipulated_data.data)
input_features = extracted_features.features

labelled_windows = LabelledWindows(input_features, manipulated_data.manipulation_indeces, window_size=40)
X = labelled_windows.windows
y = labelled_windows.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#train_set = SpikingDataset(data=X_train, targets=y_train, encoding=input_encoding, num_steps=opt.num_steps)
#test_set = SpikingDataset(data=X_test, targets=y_test, encoding=input_encoding, num_steps=opt.num_steps)

#train_set = SpikingDataset(data=X_train, targets=y_train, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=False)
#test_set = SpikingDataset(data=X_test, targets=y_test, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=False)

train_set = CustomDataset(data=X_train, targets=y_train, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=opt.flatten, set_type=opt.set_type)
test_set = CustomDataset(data=X_test, targets=y_test, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=opt.flatten, set_type=opt.set_type)
val_set = CustomDataset(data=X_val, targets=y_val, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=opt.flatten, set_type=opt.set_type)

print("Train set size: ", train_set.db[0][0].size())

print("\nClass counts: ")
print("\t- 0:", train_set.n_samples_per_class[0])
print("\t- 1:", train_set.n_samples_per_class[1])

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True)

for batch_idx, (data, target) in enumerate(train_loader):
  input_size = np.shape(data)[2]
  break

if opt.output_decoding=="rate":
  #loss_fn = mse_count_loss(correct_rate=1, incorrect_rate=0.1)
  loss_fn = functional.loss.mse_count_loss(correct_rate=1, incorrect_rate=0.2)
elif opt.output_decoding=="latency":
  loss_fn = functional.loss.mse_temporal_loss(target_is_time=False, on_target=0, off_target=opt.num_steps, tolerance=5, multi_spike=False)
else:
  raise Exception("Only rate and latency encodings allowed") 

#model = SNN(input_size=input_size, hidden_size=opt.hidden_size, output_size=2).to(device)
model = CSNN(batch_size=opt.batch_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

def train(model,train_loader,optimizer,epoch):
  model.train()
  loss_val=0
  for batch_idx, (data, target) in enumerate(train_loader):
      data=data.to(device)
      target=target.to(device)

      spk_recs, _ = model(data, opt.num_steps)
      #spk_rec = spk_recs[-1]
      spk_rec = spk_recs

      loss = loss_fn(spk_rec,target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_val+=loss.item()
      if opt.wandb_logging:
        wandb.log({"loss": loss.item()})

  loss_val /= len(train_loader)
  print("\n--------------------------------------")
  print("Epoch:\t\t",epoch)
  print("Train loss:\t%.2f" % loss_val)

  forward_pass_eval(model, val_loader)
  
  return model

def forward_pass_eval(model,dataloader, testing=False):
  model.eval()
  test_loss = 0
  acc = 0

  y_true = []
  y_pred = []

  label_names = ["No manipulation", "Manipulation"]

  with torch.no_grad():
    for data, target in dataloader:
        data=data.to(device)
        target=target.to(device)
        spk_recs, _ = model(data, opt.num_steps)
        #spk_rec = spk_recs[-1] # get spikes from last layer
        spk_rec = spk_recs
        if opt.decoding=="rate":
          acc_val = functional.acc.accuracy_rate(spk_rec,target)
        elif opt.decoding=="latency":
          acc_val = functional.acc.accuracy_temporal(spk_rec,target)

        test_loss_current = loss_fn(spk_rec,target).item()
        test_loss+=test_loss_current
        acc+=acc_val

        _, predicted = spk_rec.sum(dim=0).max(dim=1)

        if opt.wandb_logging and not testing:
          wandb.log({"Validation loss": test_loss_current, "Validation Accuracy": acc_val})

        elif opt.wandb_logging and testing:
          wandb.log({"Test loss": test_loss_current, "Test Accuracy": acc_val})

        if len(data)==opt.batch_size and testing:
          for i in range(opt.batch_size):
            y_pred.append(predicted[i].item())
            y_true.append(target[i].item())

  if testing:
    plot_confusion_matrix(y_pred, y_true)

  test_loss /= len(test_loader)
  acc /= len(test_loader)

  if testing:
    print("Test loss:\t%.4f" % test_loss)
    print("Test acc:\t%.4f" % acc,"\n")
  else:
    print("Val loss:\t%.4f" % test_loss)
    print("Val acc:\t%.4f" % acc,"\n")

def plot_confusion_matrix(y_pred, y_true):
    
    cm_prec = confusion_matrix(y_true, y_pred, labels=[0,1], normalize="pred")
    cm_sens = confusion_matrix(y_true, y_pred, labels=[0,1], normalize="true")
    conf_matrix = cm_sens*cm_prec*2/(cm_sens+cm_prec+1e-8)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, alpha=0.5, cmap=plt.cm.Blues)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('F1-Score Matrix', fontsize=18)
    plt.show()

for epoch in range(1, opt.num_epochs+1):
  model=train(model,train_loader,optimizer,epoch)

forward_pass_eval(model, test_loader test=True)
with torch.no_grad():
  test = RasterPlot(model, test_loader, opt.num_steps, device)