import torch
from torch._C import Value
import torch.nn as nn
from snntorch import spikegen

dtype = torch.float

class LossFunctions:
    def _prediction_check(self, spk_out):
        device = spk_out.device

        num_steps = spk_out.size(0)
        num_outputs = spk_out.size(-1)

        return device, num_steps, num_outputs

class mse_count_loss(LossFunctions):
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
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        loss_fn = nn.MSELoss()

        # generate ideal spike-count in C sized vector
        on_target = int(num_steps * self.correct_rate)
        off_target = int(num_steps * self.incorrect_rate)
        spike_count_target = spikegen.targets_convert(
            targets,
            num_classes=num_outputs,
            on_target=on_target,
            off_target=off_target,
        )

        spike_count = torch.sum(spk_out, 0)  # B x C

        #loss = loss_fn(spike_count, spike_count_target)
        loss = torch.mean(torch.square(spike_count_target - spike_count) * self.class_weights)
        return loss / num_steps