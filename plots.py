import torch
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
def plot_spike_trains(model, test_loader, num_steps, device, save_name, save=False):
    """
    Plots spike trains of a trained model.
    :param model: trained model
    :param test_loader: test loader
    :param num_steps: number of simulation steps
    :param device: device
    :param save_name: name of saved plot
    :param save: save plot
    """

    with torch.no_grad():
        spk_recs = get_spike_data(model, test_loader, num_steps, device)
    fig = plt.figure(facecolor="w", figsize=(5, 15))
    for i in range(len(spk_recs)):
        spk_rec = spk_recs[i]
        add_subplot(spike_data=spk_rec, fig=fig, subplt_num=int(str(len(spk_recs))+str(1)+str(i + 1)))

    fig.tight_layout(pad=2)
    if save:
        plt.savefig("plots/spike-trains_"+save_name+".png")
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
def plot_manipulated_data(manipulated_data, save=False):
    """
    Plots LOBSTER data with manipulations highighted.
    :param manipulated_data: manipulated
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plot_range = [0, 1400]
    for y in range(2):
            for i in range(plot_range[0], plot_range[1]):
                if i in manipulated_data.manipulation_indeces:
                    axs[y].axvline(x = i-plot_range[0], color = 'r', label = 'axvline - full height', linestyle='dotted', linewidth=0.5)
                axs[y].plot(manipulated_data.data[y][plot_range[0]:plot_range[1]], color='b', linewidth=1)

    if save:
        plt.savefig("plots/manipulated-data.png")
    plt.show()
    

"""
#######################
plot manipulated window
#######################
"""
def plot_manipulated_window(data, targets, save=False):
    """
    Plots only one window containing a mnipulation.
    """
    for i in range(len(data)):
        if targets[i] == 1:
            print("Manipulated sample:", i)
            plt.plot(data[i])
            break
    
    if save:
        plt.savefig("plots/manipulated-window.png")
    plt.show()

"""
#####################
plot confusion matrix
#####################
"""
def plot_confusion_matrix(y_pred, y_true, save_name, save=False):
    """
    Plots confusion matrix and normalized confusion matrix.
    :param y_pred: predicted labels
    :param y_true: true labels
    :param save_name: name of saved plot
    :param save: save plot
    """

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0,1])
    
    normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    normalized_conf_matrix = np.round(normalized_conf_matrix, decimals=4)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5))

    ax[0].matshow(conf_matrix)
    ax[1].matshow(normalized_conf_matrix)

    """ax[0] = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])
    ax[1] = ConfusionMatrixDisplay(confusion_matrix = normalized_conf_matrix, display_labels = [False, True])"""

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax[0].text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large', color= 'black' if conf_matrix[i, j] > conf_matrix.max() / 2. else 'white')
            ax[1].text(x=j, y=i,s=normalized_conf_matrix[i, j], va='center', ha='center', size='xx-large', color= 'black' if normalized_conf_matrix[i, j] > normalized_conf_matrix.max() / 2. else 'white')
    
    ax[0].set_xlabel('Predictions', fontsize=18)
    ax[0].set_ylabel('Targets', fontsize=18)
    ax[0].set_title('Confusion Matrix', fontsize=22)

    ax[1].set_xlabel('Predictions', fontsize=18)
    ax[1].set_ylabel('Targets', fontsize=18)
    ax[1].set_title('Normalized Confusion Matrix', fontsize=22)

    if save:
        plt.savefig("plots/conf-matrix_"+save_name+".png")
    plt.show()

def plot_real_time(neuron1, neuron2):
    """
    Plots real time eval data.
    :param neuron1: neuron 1 spike data
    :param neuron2: neuron 2 spike data
    """
    
    neuron1_avg = []
    neuron2_avg = []
    for i in range(100, len(neuron1)+100, 100):
        for j in range(100):
            neuron1_avg.append(np.mean(neuron1[i-50:i+50]))
            neuron2_avg.append(np.mean(neuron2[i-50:i+50]))

    plt.plot(neuron1_avg, label="neuron 1")
    plt.plot(neuron2_avg, label="neuron 2")
    plt.show()