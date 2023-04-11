# local imports
from data import LobsterData
"""from construct_dataset import ManipulatedDataset
from construct_dataset import ExtractFeatures
from construct_dataset import LabelledWindows
from construct_dataset import CustomDataset"""
import construct_dataset
from construct_dataset import CustomDataset
from model import SNN, CSNN, CNN, CSNNGaussian
from options import Options
from plots import RasterPlot, ManipulationPlot
from metrics import Metrics
from early_stopping import EarlyStopping
import sys

# external imports
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import wandb

opt = Options().parse()

if opt.wandb_logging:
    
    print("\n#########################################")
    print("#########################################\n")
    print("!!IMPORTANT!! You need to create a WandB account and paste your atuhorization key in options.py to use this.")
    print("\n#########################################")
    print("#########################################\n")

    key=opt.wandb_key
    wandb.login(key=key)
    wandb.init(project=opt.wandb_project, 
               entity=opt.wandb_entity, 
               group=opt.wandb_group,
               settings=wandb.Settings(start_method="thread"),
               config=opt)

unmanipulated_data = LobsterData()

if opt.train_method == "multiclass":
  X, y = construct_dataset.prepare_data(unmanipulated_data.orderbook_data, True, opt.window_length, opt.window_overlap)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
else:

  X_train, y_train = construct_dataset.prepare_data(unmanipulated_data.orderbook_data, False, opt.window_length, opt.window_overlap)
  X_test, y_test = construct_dataset.prepare_data(unmanipulated_data.orderbook_data, True, opt.window_length, opt.window_overlap)


  X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
  #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

  X_temp, X_test, y_temp, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
  _, X_val, _, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

train_set = CustomDataset(data=X_train, targets=y_train, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=opt.flatten_data, set_type=opt.set_type)
test_set = CustomDataset(data=X_test, targets=y_test, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=opt.flatten_data, set_type=opt.set_type)
val_set = CustomDataset(data=X_val, targets=y_val, encoding=opt.input_encoding, num_steps=opt.num_steps, flatten=opt.flatten_data, set_type=opt.set_type)

print("\n----------------------------------\n")
print("Class counts: ")
print("\t- 0:", train_set.n_samples_per_class[0])
print("\t- 1:", train_set.n_samples_per_class[1])
print("\n----------------------------------\n")

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True)

print("Train set size:\t\t", len(train_set))
print("Test set size:\t\t", len(test_set))
print("Validation set size:\t", len(val_set))
print("\n----------------------------------\n")

for batch_idx, (data, target) in enumerate(train_loader):
  input_size = np.shape(data)[2]
  break

metrics = Metrics(opt.net_type, opt.set_type, opt.output_decoding, train_set, opt.device, opt.num_steps)
early_stopping = EarlyStopping(patience=10, verbose=True)

if opt.net_type=="CSNN":
  model = CSNNGaussian(batch_size=opt.batch_size).to(opt.device)
  if opt.train_method == "oneclass":
    model.load_state_dict(torch.load('checkpoint.pt')) # load best model
    model.freeze_body()
    model.reset_head()
elif opt.net_type=="SNN":
  model = SNN(input_size=input_size, hidden_size=opt.hidden_size, output_size=2).to(opt.device)
elif opt.net_type=="CNN":
  model = CNN().to(opt.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))


def train(model,train_loader,optimizer,epoch, logging_index, stop_early):
  model.train()
  loss_val=0
  print("\nEpoch:\t\t",epoch,"/",opt.num_epochs)
  for batch_idx, (data, target) in enumerate(train_loader):
      if stop_early:
        break
      data=data.to(opt.device)
      target=target.to(opt.device)

      if opt.train_method == "oneclass" and batch_idx % 2 == 0:
        data = model.generate_gaussian_feature(opt.batch_size, opt.num_steps).to(opt.device)
        target = torch.ones(opt.batch_size, dtype=torch.long).to(opt.device)
        output = metrics.forward_pass(model, data, opt.num_steps, gaussian=True)

      else:
        output = metrics.forward_pass(model, data, opt.num_steps)

      loss = metrics.loss(output,target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_val+=loss.item()
      if opt.wandb_logging:
        wandb.log({"Train loss": loss.item(),
                   "Train index": logging_index})

      logging_index+=1

  loss_val /= len(train_loader)
  print("Train loss:\t%.3f" % loss_val)

 
  return model, logging_index

def forward_pass_eval(model,dataloader, logging_index, testing=False):
  model.eval()
  loss = 0
  acc = 0
  stop_early = False

  y_true = []
  y_pred = []

  label_names = ["No manipulation", "Manipulation"]

  if testing:
    print("loading best model...")
    model.load_state_dict(torch.load('checkpoint.pt')) # load best model

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(dataloader):
      data=data.to(opt.device)
      target=target.to(opt.device)

      """if opt.train_method == "oneclass" and batch_idx % 2 == 0:
        data = model.generate_gaussian_feature(opt.batch_size, opt.num_steps).to(opt.device)
        target = torch.ones(opt.batch_size, dtype=torch.long).to(opt.device)
        output = metrics.forward_pass(model, data, opt.num_steps, gaussian=True)

      else:"""
      output = metrics.forward_pass(model, data, opt.num_steps)

      #output = metrics.forward_pass(model, data, opt.num_steps)
      acc_current = metrics.accuracy(output,target)

      loss_current = metrics.loss(output,target).item()
      loss+=loss_current
      acc+=acc_current

      if opt.wandb_logging and not testing:
        wandb.log({"Validation loss": loss_current, 
                    "Validation Accuracy": acc_current,
                    "Validation index": logging_index})

      elif opt.wandb_logging and testing:
        wandb.log({"Test loss": loss_current, 
                    "Test Accuracy": acc_current,
                    "Test index": logging_index})

      predicted = metrics.return_predicted(output)
      if len(data)==opt.batch_size and testing:
        for i in range(opt.batch_size):
          y_pred.append(predicted[i].item())
          y_true.append(target[i].item())

      logging_index+=1

  if testing:
    plot_confusion_matrix(y_pred, y_true)

    loss /= len(test_loader)
    acc /= len(test_loader)
    acc = acc*100

  else:
    loss /= len(val_loader)
    acc /= len(val_loader)
    acc = acc*100
    
  
  if testing:
    print("Test loss:\t%.3f" % loss)
    print("Test acc:\t%.2f" % acc+"%\n")
  else:
    print("Val loss:\t%.3f" % loss)
    print("Val acc:\t%.2f"% acc+"%\n")

    early_stopping(loss, model)

    if early_stopping.early_stop:
      print("Early stopping")
      stop_early = True

    return logging_index, stop_early

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

logging_index_train = 0
logging_index_forward_eval = 0
stop_early = False

for epoch in range(1, opt.num_epochs+1):

  model, logging_index_train = train(model,train_loader,optimizer,epoch, logging_index_train, stop_early)
  logging_index_forward_eval, stop_early = forward_pass_eval(model, val_loader, logging_index_forward_eval)

logging_index_forward_eval = 0
forward_pass_eval(model, test_loader, logging_index_forward_eval, testing=True)

RasterPlot(model, test_loader, opt.num_steps, opt.device)

wandb.finish()