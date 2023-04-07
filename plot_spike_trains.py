import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from snntorch import spikeplot
import numpy as np

class RasterPlot(object):
    def __init__(self, model, test_loader, num_steps, device):

        spk_recs = self.get_spike_data(model, test_loader, num_steps, device)
        self.plot(spk_recs)

    def plot(self, spk_recs):
        fig = plt.figure(facecolor="w", figsize=(10, 15))
        for i in range(len(spk_recs)):
            spk_rec = spk_recs[i]
            self.add_subplot(spike_data=spk_rec, fig=fig, subplt_num=int(str(len(spk_recs))+str(1)+str(i + 1)))

        fig.tight_layout(pad=2)
        plt.show()


    def get_spike_data(self, model, test_loader, num_steps, device):
        model.eval()

        #normal testing data
        test_data, test_target = next(iter(test_loader))
        test_data=test_data.to(device) # only one element in batch
        test_target=test_target.to(device) # only one element in batch

        spk_recs, _ = model(test_data, num_steps)
        
        for i in range(len(spk_recs)):
            spk_recs[i] = torch.permute(spk_recs[i],(1,0,2))[0]

        spk_recs.insert(0, test_data[0])

        return spk_recs

    def add_subplot(self,spike_data,fig,subplt_num):
        spikeplot.raster(spike_data, fig.add_subplot(subplt_num), s=1.5, c="blue")

        ax = fig.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))



