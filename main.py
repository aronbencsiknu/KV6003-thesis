# local imports
from data import LobsterData
import construct_dataset
from construct_dataset import CustomDataset
from receptive_encoder import CUBALayer
from model import SNN, CSNN, CNN, OC_SCNN, RNN
from options import Options
import plots
from metrics import Metrics
from early_stopping import EarlyStopping
from sweep import SweepHandler

# external imports
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb
import pathlib
from snntorch import surrogate
import json

# parse arguments
opt = Options().parse()

if opt.wandb_logging or opt.sweep:
  print("\n#########################################")
  print("#########################################\n")
  print("!!IMPORTANT!! You need to create a WandB account and paste your authorization key in options.py to use this.")
  print("\n#########################################")
  print("#########################################\n")

# init wandb logging if specified
if opt.wandb_logging:
  key=opt.wandb_key
  wandb.login(key=key)
  wandb.init(project=opt.wandb_project, 
              entity=opt.wandb_entity, 
              group=opt.run_name,
              settings=wandb.Settings(start_method="thread"),
              config=opt)

# init sweep handler if specified
if opt.sweep:
  sweep_handler = SweepHandler()

  key=opt.wandb_key
  wandb.login(key=key)

  sweep_config = {'method': 'random'}
  sweep_config['metric'] = sweep_handler.metric
  sweep_config['parameters'] = sweep_handler.parameters_dict

  sweep_id = wandb.sweep(sweep_config, project="thesis_sweeps_final")

if opt.train_method == "multiclass":
  oneclass = False
  train_data = LobsterData(path="Amazon", limit=40000)
  test_data = LobsterData(path="Apple", limit=20000)

  X_train, y_train, means = construct_dataset.prepare_data(train_data.orderbook_data, True, opt.window_length, opt.window_overlap, opt.manipulation_length, opt.subset_indeces, 0.11)
  X_test, y_test, _ = construct_dataset.prepare_data(test_data.orderbook_data, True, opt.window_length, opt.window_overlap, opt.manipulation_length, opt.subset_indeces, 0.11)

  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.1, random_state=42)

else:
  oneclass = True
  train_data = LobsterData(path="Amazon")
  test_data = LobsterData(path="Apple")

  if not opt.load_model:
    X_train, y_train, _ = construct_dataset.prepare_data(train_data.orderbook_data, False, opt.window_length, opt.window_overlap, opt.manipulation_length)
  X_test, y_test, _ = construct_dataset.prepare_data(test_data.orderbook_data, True, opt.window_length, opt.window_overlap, opt.manipulation_length)

  X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
  X_temp, X_test, y_temp, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
  _, X_val, _, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

# get feature dimensionality
feature_dimensionality = np.shape(X_train)[2]

# create receptive encoder if encoding is population
if opt.input_encoding == "population":
  if opt.load_model:
    gain_name = opt.load_name+".npy"
    gains = np.load(pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / gain_name))
    print(gains)
  else:
    gains = None

  receptive_encoder = CUBALayer(feature_dimensionality=feature_dimensionality, population_size=10, means=means, predefined_gains=None)
  opt.num_steps = int(receptive_encoder.T/receptive_encoder.dt) * opt.window_length # override num_steps
  receptive_encoder.display_tuning_curves() # plot tuning curves

else:
  receptive_encoder = None

# create datasets and dataloaders
train_set = CustomDataset(data=X_train, targets=y_train, num_steps=opt.num_steps, window_length=opt.window_length, encoding=opt.input_encoding, flatten=opt.flatten_data, set_type=opt.set_type, pop_encoder=receptive_encoder)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

test_set = CustomDataset(data=X_test, targets=y_test, num_steps=opt.num_steps, window_length=opt.window_length, encoding=opt.input_encoding, flatten=opt.flatten_data, set_type=opt.set_type, pop_encoder=receptive_encoder)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True)

val_set = CustomDataset(data=X_val, targets=y_val, num_steps=opt.num_steps, window_length=opt.window_length, encoding=opt.input_encoding, flatten=opt.flatten_data, set_type=opt.set_type, pop_encoder=receptive_encoder)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True)

# get input size (num features for non-flattened data; num of datapoints in window for flattened data)
data, _ = next(iter(test_loader))
input_size = np.shape(data)[2]

# print train/test/valid set info
print("\n----------------------------------\n")
print("Class counts in train set: ")
print("\t- 0:", train_set.n_samples_per_class[0])
print("\t- 1:", train_set.n_samples_per_class[1])
print("Class counts in test set: ")
print("\t- 0:", test_set.n_samples_per_class[0])
print("\t- 1:", test_set.n_samples_per_class[1])
print("\n----------------------------------\n")
print("Train set size:\t\t", len(train_set))
print("Test set size:\t\t", len(test_set))
print("Validation set size:\t", len(val_set))
print("\n----------------------------------\n")

if opt.load_model:
  metrics = Metrics(opt.net_type, opt.set_type, opt.output_decoding, test_set, opt.device, opt.num_steps, oneclass)
else:
  metrics = Metrics(opt.net_type, opt.set_type, opt.output_decoding, train_set, opt.device, opt.num_steps, oneclass)

# initialize early stopping
early_stopping = EarlyStopping(patience=20, verbose=True)

# Select model
if opt.train_method == "oneclass":
  metrics.net_type = "OC_SCNN"
  opt.net_type = "OC_SCNN"
  model = OC_SCNN(batch_size=opt.batch_size, input_size=feature_dimensionality).to(opt.device)

  path = pathlib.Path("trained_models") / "oneclass.pt"
  model.load_state_dict(torch.load(path))

  model.freeze_body() # freeze feature extraction layers
  model.reset_head() # reset head layers

elif opt.net_type=="OC_SCNN":
  model = OC_SCNN(batch_size=opt.batch_size, input_size=feature_dimensionality).to(opt.device)

elif opt.net_type=="CSNN":
  model = CSNN(batch_size=opt.batch_size, input_size=feature_dimensionality).to(opt.device)
  
elif opt.net_type=="SNN":

  # load hyperparameter dictionary if specified
  if opt.load_model_dict:
    name_dict = opt.load_name + ".json"
    path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / name_dict)

    p_dict = open(path_p_dict)
    p_dict = json.load(p_dict)
    model = SNN(input_size=input_size, hidden_size=opt.hidden_size, output_size=2, h_params=p_dict).to(opt.device)
  else:  
    model = SNN(input_size=input_size, hidden_size=opt.hidden_size, output_size=2).to(opt.device)

elif opt.net_type=="CNN":
  model = CNN(input_size=feature_dimensionality).to(opt.device)
elif opt.net_type=="RNN":
  model = RNN(input_size=feature_dimensionality).to(opt.device)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

def train_epoch(net, train_loader,optimizer,epoch, logging_index):

  net.train()
  loss_val=0

  print("\nEpoch:\t\t",epoch,"/",opt.num_epochs)
  for batch_idx, (data, target) in enumerate(train_loader):
      data=data.to(opt.device)
      target=target.to(opt.device)

      if opt.train_method == "oneclass" and batch_idx % 2 == 0:
        
        data = net.generate_gaussian_feature(opt.batch_size, opt.num_steps).to(opt.device)
        target = torch.ones(opt.batch_size, dtype=torch.long).to(opt.device)
        output = metrics.forward_pass(net, data, opt.num_steps, gaussian=True)

      else:
        output = metrics.forward_pass(net, data, opt.num_steps)

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
  print("Train loss:\t%.6f" % loss_val)
 
  return net, logging_index, loss_val

def sweep_train(config=None):
    
    # Initialize a new wandb run
    with wandb.init(config=config):
        
      # If called by wandb.agent, as below,
      # this config will be set by Sweep Controller
      config = wandb.config

      hidden_size = [config.hidden_size]*config.num_hidden
      
      # initialize surrogate gradient
      if config.surrogate_gradient == "atan":
        spike_grad = surrogate.atan()

      elif config.surrogate_gradient == "sigmoid":
        spike_grad = surrogate.sigmoid()
      
      elif config.surrogate_gradient == "fast_sigmoid":
        spike_grad = surrogate.fast_sigmoid()

      # replace previously defined network with the one initialized by wandb sweep
      network = SNN(input_size=input_size, hidden_size=hidden_size, dropout=config.dropout, neuron_type=config.NeuronType, learn_alpha=config.learn_alpha, learn_beta=config.learn_beta, learn_threshold=config.learn_threshold, spike_grad=spike_grad).to(opt.device)
      
      optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
      sweep_early_stopping = EarlyStopping(patience=opt.num_epochs, verbose=True)
      max_acc = 0

      for epoch in range(opt.num_epochs):
          network, _, _ = train_epoch(network, train_loader, optimizer, epoch, logging_index=0)
          _, _, avg_loss, avg_acc = forward_pass_eval(network, val_loader, sweep_early_stopping, logging_index=0)
          if avg_acc > max_acc:
            max_acc = avg_acc
          wandb.log({"loss": avg_loss, "epoch": epoch, "accuracy": avg_acc}) 


      config_dict = config.__dict__
      config_dict = config_dict["_items"]

      dict_name = str(max_acc)+ "_" + opt.run_name + ".json"
      path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / dict_name)
     
      with open(path_p_dict, 'w') as f:
        json.dump(config_dict, f)
      
      
      model_name = str(max_acc)+ "_" + opt.run_name + ".pt"
      
      path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / model_name)
      network.load_state_dict(torch.load('checkpoint.pt')) # load best model
      torch.save(network.state_dict(), path_model)

def load_model():
  name_model = opt.load_name + "_model.pt"
  name_dict = opt.load_name + "_dict.json"
  path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / name_model)
  path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / name_dict)

  # load hyperparameters
  p_dict = open(path_p_dict)
  p_dict = json.load(p_dict)

  # initialize model with hyperparameters and load weights
  model = SNN(input_size=input_size, hidden_size=opt.hidden_size, output_size=2, h_params=p_dict).to(opt.device)
  model.load_state_dict(torch.load(path_model, map_location=torch.device(opt.device)))

  return model

def forward_pass_eval(model,dataloader, early_stopping, logging_index, testing=False):

  model.eval()
  loss = 0
  acc = 0
  stop_early = False

  y_true = []
  y_pred = []

  if testing:
    print("loading best model...")

    if opt.load_model:
      model = load_model()
    else:
      model.load_state_dict(torch.load('checkpoint.pt')) # load best model

      if opt.save_model:
        model_name = opt.run_name + ".pt"
        path = pathlib.Path(pathlib.Path.cwd() / "trained_models") / model_name
        torch.save(model.state_dict(), path)

        # save gains
        if opt.input_encoding=="population":
            gains = []
            for population in receptive_encoder.populations:
              gains.append(population.gains)
            gains = np.array(gains)
            gains_name ="gain_test.npy"
            path = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / gains_name)
            np.save(path, gains)

      if opt.net_type=="OC_SCNN" and opt.train_method=="multiclass":
        path = pathlib.Path(pathlib.Path.cwd() / "trained_models") / "oneclass.pt"
        torch.save(model.state_dict(), path)

  with torch.no_grad():

    for batch_idx, (data, target) in enumerate(dataloader):

      data=data.to(opt.device)
      target=target.to(opt.device)

      output = metrics.forward_pass(model, data, opt.num_steps)

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
    metrics.perf_measure(y_true, y_pred)
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
    print("Precision:\t%.3f" % metrics.precision())
    print("Recall:\t%.3f"% metrics.recall())
    print("F1-Score:\t%.3f"% metrics.f1_score() +"\n")

    if opt.save_results:
      with open('results/'+opt.run_name+'.txt', 'w') as file:

        # Write the values to the text file
        file.write("\nTest loss: "+str(loss))
        file.write("\nTest acc: "+str(acc))
        file.write("\nPrecision: "+str(metrics.precision()))
        file.write("\nRecall: "+str(metrics.recall()))
        file.write("\nF1-Score: "+str(metrics.f1_score()))

        file.close()

    plots.plot_confusion_matrix(y_pred, y_true, opt.run_name, opt.save_results)

  else:
    print("Val loss:\t%.3f" % loss)
    print("Val acc:\t%.3f"% acc+"%\n")
    early_stopping(loss, model)

    if early_stopping.early_stop:
      print("Early stopping")
      stop_early = True

    return logging_index, stop_early, loss, acc

def real_time_eval(model, dataloader):
  model = load_model()
  real_time_spk = []
  neuron1 = []
  neuron2 = []

  for batch_idx, (data, target) in enumerate(dataloader):
    data=data.to(opt.device)
    target=target.to(opt.device)

    if data.shape[0] == opt.batch_size:
      for i in range(opt.batch_size):
        if target[i] == 0:
          output = metrics.forward_pass(model, data[i], opt.num_steps, time_first=True, real_time=False)

          output = output[0][-1]

          for j in range(opt.num_steps):
            neuron1.append(output[j][0].item())
            neuron2.append(output[j][1].item())

  plots.plot_real_time(neuron1, neuron2)
  
  return real_time_spk

logging_index_train = 0
logging_index_forward_eval = 0
stop_early = False

if not opt.load_model and not opt.sweep and not opt.real_time:
  for epoch in range(1, opt.num_epochs+1):

    model, logging_index_train, _ = train_epoch(model,train_loader,optimizer,epoch, logging_index_train)
    logging_index_forward_eval, stop_early, _, _ = forward_pass_eval(model, val_loader, early_stopping, logging_index_forward_eval)

    # cut off training if early stopping is triggered
    if stop_early:
      break

  logging_index_forward_eval = 0

if opt.sweep:
  wandb.agent(sweep_id, sweep_train, count=50)

elif opt.real_time:
  real_time_eval(model, test_loader)

else:
  forward_pass_eval(model, test_loader, early_stopping, logging_index_forward_eval, testing=True)

if opt.wandb_logging:
  wandb.finish()

if opt.set_type=="spiking":
  if opt.load_model:
    model = load_model()
  
  # plot spike trains
  plots.plot_spike_trains(model, test_loader, opt.num_steps, opt.device, opt.run_name, opt.save_results)