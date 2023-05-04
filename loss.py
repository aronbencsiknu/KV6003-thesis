import torch
from torch._C import Value
import torch.nn as nn
from snntorch import spikegen

dtype = torch.float

class LossFunctions:
    def _prediction_check(self, spk_out):

        num_steps = spk_out.size(0)
        num_outputs = spk_out.size(-1)

        return num_steps, num_outputs

class mse_count_loss(LossFunctions):
    """
    MSE spike count loss fromm snnTorch with class weights
    """
    def __init__(
        self,
        correct_rate=1,
        incorrect_rate=0,
        num_classes=False,
        class_weights=None
    ):
        self.correct_rate = correct_rate
        self.incorrect_rate = incorrect_rate
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.__name__ = "mse_count_loss"

    def __call__(self, spk_out, targets):
        num_steps, num_outputs = self._prediction_check(spk_out)

        # generate ideal spike-count in C sized vector
        on_target = int(num_steps * self.correct_rate)
        off_target = int(num_steps * self.incorrect_rate)
        spike_count_target = spikegen.targets_convert(
            targets,
            num_classes=num_outputs,
            on_target=on_target,
            off_target=off_target,
        )

        spike_count = torch.sum(spk_out, 0)
        if self.class_weights is not None:
            
            # pytorch MSE is replaced with custom MSE to allow for class weights
            loss = torch.mean(torch.square(spike_count_target - spike_count) * self.class_weights)
        else:
            loss = torch.mean(torch.square(spike_count_target - spike_count))

        return loss / num_steps