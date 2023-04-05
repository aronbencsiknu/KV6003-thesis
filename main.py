from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import functional
from snntorch import surrogate

from data import LobsterData
from construct_dataset import ManipulatedDataset
from construct_dataset import ExtractFeatures
from construct_dataset import LabelledWindows
from construct_dataset import SpikingDataset
from model import SNN
from options import Options
import torch
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_decoding", type=str, default="rate", help="rate or latency")
argparser.add_argument("--input_encoding", type=str, default="rate", help="rate or latency")
parsed_args = argparser.parse_args()
output_decoding = parsed_args.output_decoding
input_encoding = parsed_args.input_encoding

opt = Options()
device = opt.device

unmanipulated_data = LobsterData()
manipulated_data = ManipulatedDataset(unmanipulated_data.orderbook_data)

#plt.plot(manipulated_data.original_data[0], linestyle = '-')
#plt.show()

extracted_features = ExtractFeatures(manipulated_data.data)
input_features = extracted_features.features

labelled_windows = LabelledWindows(input_features, manipulated_data.manipulation_indeces, window_size=10)
X = labelled_windows.windows

y = labelled_windows.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_set = SpikingDataset(data=X_train, targets=y_train, encoding=input_encoding, num_steps=opt.num_steps)
test_set = SpikingDataset(data=X_test, targets=y_test, encoding=input_encoding, num_steps=opt.num_steps)

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True)


for batch_idx, (data, target) in enumerate(train_loader):
  print("Spike trains size: ", np.shape(data))
  input_size = np.shape(data)[2]
  break

if output_decoding=="rate":
  loss_fn = functional.loss.mse_count_loss(correct_rate=1, incorrect_rate=0.1)
elif output_decoding=="latency":
  loss_fn = functional.loss.mse_temporal_loss(target_is_time=False, on_target=0, off_target=opt.num_steps, tolerance=5, multi_spike=False)
else:
  raise Exception("Only rate and latency encodings allowed") 

model = SNN(input_size=input_size, hidden_size=opt.hidden_size, output_size=2).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

def train(model,train_loader,optimizer,epoch):
  model.train()
  loss_val=0
  for batch_idx, (data, target) in enumerate(train_loader):
      data=data.to(device)
      target=target.to(device)

      spk_rec, mem_rec = model(data, opt.num_steps)
      loss = loss_fn(spk_rec,target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_val+=loss.item()

  loss_val /= len(train_loader)
  print("Epoch:\t\t",epoch)
  print("Train loss:\t%.2f" % loss_val)
  return model

def test(model,test_loader,decoding):
  model.eval()
  test_loss = 0
  acc = 0
  with torch.no_grad():
      for data, target in test_loader:
          data=data.to(device)
          target=target.to(device)
          spk_rec, _ = model(data, opt.num_steps)
          if decoding=="rate":
            acc_val = functional.acc.accuracy_rate(spk_rec,target)
          elif decoding=="latency":
            acc_val = functional.acc.accuracy_temporal(spk_rec,target)
          test_loss_current = loss_fn(spk_rec,target).item()
          test_loss+=test_loss_current
          acc+=acc_val

  test_loss /= len(test_loader)
  acc /= len(test_loader)

  print("Test loss:\t%.4f" % test_loss)
  print("Test acc:\t%.4f" % acc,"\n")

for epoch in range(1, opt.num_epochs+1):
  model=train(model,train_loader,optimizer,epoch)
  test(model, test_loader, output_decoding)