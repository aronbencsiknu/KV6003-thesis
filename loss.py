import torch
from torch._C import Value
import torch.nn as nn
from snntorch import spikegen

class LossFunctions:
    def _prediction_check(self, spk_out):
        device = spk_out.device

        num_steps = spk_out.size(0)
        num_outputs = spk_out.size(-1)

        return device, num_steps, num_outputs

    def _population_code(self, spk_out, num_classes, num_outputs):
        """Count up spikes sequentially from output classes."""
        if not num_classes:
            raise Exception(
                "``num_classes`` must be specified if "
                "``population_code=True``."
            )
        if num_outputs % num_classes:
            raise Exception(
                f"``num_outputs {num_outputs} must be a factor "
                f"of num_classes {num_classes}."
            )
        device = spk_out.device
        pop_code = torch.zeros(tuple([spk_out.size(1)] + [num_classes])).to(
            device
        )
        for idx in range(num_classes):
            pop_code[:, idx] = (
                spk_out[
                    :,
                    :,
                    int(num_outputs * idx / num_classes) : int(
                        num_outputs * (idx + 1) / num_classes
                    ),
                ]
                .sum(-1)
                .sum(0)
            )
        return pop_code

class mse_count_loss(LossFunctions):
    """Mean Square Error Spike Count Loss.
    When called, the total spike count is accumulated over time for
    each neuron.
    The target spike count for correct classes is set to
    (num_steps * correct_rate), and for incorrect classes
    (num_steps * incorrect_rate).
    The spike counts and target spike counts are then applied to a
     Mean Square Error Loss Function.
    This function is adopted from SLAYER by Sumit Bam Shrestha and
    Garrick Orchard.

    Example::

        import snntorch.functional as SF

        loss_fn = SF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25)
        loss = loss_fn(outputs, targets)


    :param correct_rate: Firing frequency of correct class as a ratio, e.g.,
        ``1`` promotes firing at every step; ``0.5`` promotes firing at 50% of
        steps, ``0`` discourages any firing, defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es) as a
        ratio, e.g., ``1`` promotes firing at every step; ``0.5`` promotes
        firing at 50% of steps, ``0`` discourages any firing, defaults to ``1``
    :type incorrect_rate: float, optional

    :param population_code: Specify if a population code is applied, i.e., the
        number of outputs is greater than the number of classes. Defaults to
        ``False``
    :type population_code: bool, optional

    :param num_classes: Number of output classes must be specified if
        ``population_code=True``. Must be a factor of the number of output
        neurons if population code is enabled. Defaults to ``False``
    :type num_classes: int, optional

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(
        self,
        correct_rate=1,
        incorrect_rate=0,
        population_code=False,
        num_classes=False,
    ):
        self.correct_rate = correct_rate
        self.incorrect_rate = incorrect_rate
        self.population_code = population_code
        self.num_classes = num_classes
        self.__name__ = "mse_count_loss"

    def __call__(self, spk_out, targets, class_weights=None):
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        loss_fn = nn.NLLLoss(weight=class_weights)

        # generate ideal spike-count in C sized vector
        on_target = int(num_steps * self.correct_rate)
        off_target = int(num_steps * self.incorrect_rate)
        """spike_count_target = spikegen.targets_convert(
            targets,
            num_classes=num_outputs,
            on_target=on_target,
            off_target=off_target,
        )"""
        #_, predicted = spk_rec.sum(dim=0).max(dim=1)
        spike_count = torch.sum(spk_out, 0)  # B x C
        loss = loss_fn(spike_count, targets)
        return loss

