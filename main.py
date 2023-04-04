from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import functional
from snntorch import surrogate
from snntorch import backprop

from construct_dataset import ManipulatedDataset
from data import LobsterData
from construct_dataset import ExtractFeatures
from model import SNN
from options import Options
import torch
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_decoding", type=str, default="rate", help="rate or latency")
argparser.add_argument("--input_decoding", type=str, default="rate", help="rate or latency")
parsed_args = argparser.parse_args()
output_decoding = parsed_args.output_decoding
input_decoding = parsed_args.input_decoding

opt = Options()
device = opt.device

unmaipulated_data = LobsterData()
manipulated_data = ManipulatedDataset(unmaipulated_data.data)
extracted_features = ExtractFeatures(manipulated_data.data)

input_features = extracted_features.features

if output_decoding=="rate":
  loss_fn = functional.loss.mse_count_loss(correct_rate=1, incorrect_rate=0.1)
elif output_decoding=="latency":
  loss_fn = functional.loss.mse_temporal_loss(target_is_time=False, on_target=0, off_target=num_steps, tolerance=5, multi_spike=False)
else:
  raise Exception("Only rate and latency encodings allowed") 

model = SNN(input_size=extracted_features.num_features, hidden_size=hidden_size, output_size=extracted_features.num_classes, num_steps=num_steps, device=device).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config.lr, betas=(0.9, 0.999))

def train(model,train_loader,optimizer,epoch,index):
  model.train()
  loss_val=0
  for batch_idx, (data, target) in enumerate(train_loader):
      data=data.to(device)
      target=target.to(device)

      spk_rec, mem_rec = model(data,time_first=False)
      loss = loss_fn(spk_rec,target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_val+=loss.item()
      """wandb.log({"Train loss": loss.item(),
                       "Train index": index})"""

  loss_val /= len(train_loader)
  print("Epoch:\t\t",epoch)
  print("Train loss:\t%.2f" % loss_val)
  return model,index

def test(model,test_loader,decoding,index):
  model.eval()
  test_loss = 0
  acc = 0
  with torch.no_grad():
      for data, target in test_loader:
          data=data.to(device)
          target=target.to(device)
          spk_rec, _ = model(data, time_first=False)
          if decoding=="rate":
            acc_val = functional.acc.accuracy_rate(spk_rec,target)
          elif decoding=="latency":
            acc_val = functional.acc.accuracy_temporal(spk_rec,target)
          test_loss_current = loss_fn(spk_rec,target).item()
          test_loss+=test_loss_current
          acc+=acc_val
          """wandb.log({"Test loss": test_loss_current,
                    "Test Accuracy": acc_val,
                          "Test index": index})"""
          index+=1

  test_loss /= len(test_loader)
  acc /= len(test_loader)

  print("Test loss:\t%.2f" % test_loss)
  print("Test acc:\t%.2f" % acc,"\n")

  return index