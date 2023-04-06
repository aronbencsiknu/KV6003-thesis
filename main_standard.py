from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import functional
from snntorch import surrogate

from data import LobsterData
from construct_dataset import ManipulatedDataset
from construct_dataset import ExtractFeatures
from construct_dataset import LabelledWindows
from construct_dataset import NonSpikingDataset
from loss import mse_count_loss
from model import SNN, CNN
from options import Options
import torch
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

opt = Options()
device = opt.device

print("\nConstructing dataset...")
unmanipulated_data = LobsterData()

manipulated_data = ManipulatedDataset(unmanipulated_data.orderbook_data)

extracted_features = ExtractFeatures(manipulated_data.data)
input_features = extracted_features.features

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(extracted_features.features[0][:500])
axs[0].ticklabel_format(style='plain') 
for i in range(500):
  if i in manipulated_data.manipulation_indeces:
    axs[0].axvline(x = i, color = 'r', label = 'axvline - full height')
axs[1].plot(extracted_features.features[1][:500])
plt.show()

labelled_windows = LabelledWindows(input_features, manipulated_data.manipulation_indeces, window_size=30)
X = labelled_windows.windows
y = labelled_windows.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_set = NonSpikingDataset(data=X_train, targets=y_train)
test_set = NonSpikingDataset(data=X_test, targets=y_test)

print("class counts: ", train_set.n_samples_per_class)

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True)

model = CNN().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

loss_fn = torch.nn.NLLLoss(train_set.weights4balance().to(device))

def train(model,train_loader,optimizer,epoch):
  model.train()
  loss_val=0
  for batch_idx, (data, target) in enumerate(train_loader):

    data=data.to(device)
    target=target.to(device)
    """if target[0] == 1:
       fig, axs = plt.subplots(1, 2, figsize=(10, 5))
       axs[0].plot(data[0][0])
       axs[1].plot(data[0][1])
       plt.show()"""
    output = model(data)
    loss = loss_fn(output,target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val+=loss.item()

  loss_val /= len(train_loader)
  print("\n--------------------------------------")
  print("Epoch:\t\t",epoch)
  print("Train loss:\t%.2f" % loss_val)
  return model


def test(model,test_loader):
    model.eval()
    test_loss = 0
    acc = 0
    test_accuracy = 0
    y_true = []
    y_pred = []

    label_names = ["No manipulation", "Manipulation"]

    with torch.no_grad():
        for data, target in test_loader:
            data=torch.squeeze(data).to(device)
            
            target=target.to(device)
            output = model(data)

            
            ps = torch.exp(output)
            top_class = torch.argmax(ps, dim=1)
            print(top_class)
            equals = top_class == target
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if len(data)==opt.batch_size:
                for i in range(opt.batch_size):
                  y_pred.append(top_class[i])
                  y_true.append(target[i].item())
    
    test_accuracy = test_accuracy/len(test_loader)
    print(f"\n\nTest accuracy: {test_accuracy:.3f}\n")


    plot_confusion_matrix(y_pred, y_true)

def plot_confusion_matrix(y_pred, y_true):
    
    cm_prec = confusion_matrix(y_true, y_pred, labels=[0,1], normalize="pred")
    cm_sens = confusion_matrix(y_true, y_pred, labels=[0,1], normalize="true")
    conf_matrix = cm_sens*cm_prec*2/(cm_sens+cm_prec+1e-8)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('F1-Score Matrix', fontsize=18)
    plt.show()

for epoch in range(1, opt.num_epochs+1):
  model=train(model,train_loader,optimizer,epoch)

test(model, test_loader)
