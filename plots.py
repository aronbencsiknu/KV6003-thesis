import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from snntorch import spikeplot
import numpy as np
from sklearn.metrics import confusion_matrix


"""
#################
plot spike trains
#################
"""
def plot_spike_trains(model, test_loader, num_steps, device):
    with torch.no_grad():
        spk_recs = get_spike_data(model, test_loader, num_steps, device)
    fig = plt.figure(facecolor="w", figsize=(10, 15))
    for i in range(len(spk_recs)):
        spk_rec = spk_recs[i]
        add_subplot(spike_data=spk_rec, fig=fig, subplt_num=int(str(len(spk_recs))+str(1)+str(i + 1)))

    fig.tight_layout(pad=2)
    plt.show()

def get_spike_data(model, test_loader, num_steps, device):
    model.eval()

    # normal testing data
    test_data, test_target = next(iter(test_loader))
    test_data=test_data.to(device) # only one element in batch
    test_target=test_target.to(device) # only one element in batch

    spk_recs, _ = model(test_data, num_steps)
    
    for i in range(len(spk_recs)):
        spk_recs[i] = torch.permute(spk_recs[i],(1,0,2))[0]

    spk_recs.insert(0, test_data[0])
    return spk_recs

def add_subplot(spike_data,fig,subplt_num):
    spikeplot.raster(spike_data, fig.add_subplot(subplt_num), s=1.5, c="blue")

    ax = fig.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


"""
#################################################
plot data with injected manipulations highlighted
#################################################
"""
def plot_manipulated_data(manipulated_data):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plot_range = [0, 1400]
    for y in range(2):
            for i in range(plot_range[0], plot_range[1]):
                if i in manipulated_data.manipulation_indeces:
                    axs[y].axvline(x = i, color = 'r', label = 'axvline - full height', linestyle='dotted', linewidth=0.5)
                axs[y].plot(manipulated_data.data[y][plot_range[0]:plot_range[1]], color='b', linewidth=1)

    plt.show()

"""
#######################
plot manipulated window
#######################
"""
def plot_manipulated_window(data, targets):
    for i in range(len(data)):
        if targets[i] == 1:
            print("Manipulated sample:", i)
            plt.plot(data[i])
            break

    plt.show()

"""
#####################
plot confusion matrix
#####################
"""
def plot_confusion_matrix(y_pred, y_true):

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0,1])
    
    normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    normalized_conf_matrix = np.round(normalized_conf_matrix, decimals=2)
    

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5))

    ax[0].matshow(conf_matrix, alpha=0.5, cmap=plt.cm.Blues)
    ax[1].matshow(normalized_conf_matrix, alpha=0.5, cmap=plt.cm.Blues)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax[0].text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
            ax[1].text(x=j, y=i,s=normalized_conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    ax[0].set_xlabel('Predictions', fontsize=18)
    ax[0].set_ylabel('Targets', fontsize=15)
    ax[0].set_title('Confusion Matrix', fontsize=15)

    ax[1].set_xlabel('Predictions', fontsize=18)
    ax[1].set_ylabel('Targets', fontsize=15)
    ax[1].set_title('Normalized Confusion Matrix', fontsize=15)

    plt.show()