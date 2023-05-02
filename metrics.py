from snntorch import functional
import torch
from loss import mse_count_loss

class Metrics():
    def __init__(self, net_type, set_type, output_decoding, train_set, device, num_steps, oneclass=False):
        self.net_type = net_type
        self.set_type = set_type
        self.output_decoding = output_decoding
        self.train_set = train_set
        self.oneclass = oneclass
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        self.spiking = True if self.net_type=="CSNN" or self.net_type=="SNN" or net_type == "OC_SCNN" else False
        
        if self.oneclass or self.train_set is None:
            class_weights = None
        else:
            class_weights = train_set.get_class_weights().to(device)

        if self.set_type=="non-spiking":
            print("Non-spiking network")
            self.loss_fn = torch.nn.NLLLoss(class_weights)

        elif self.output_decoding=="rate":
            self.loss_fn = mse_count_loss(correct_rate=1.0, incorrect_rate=0.1, class_weights=class_weights)

        elif self.output_decoding=="latency":
            self.loss_fn = functional.loss.mse_temporal_loss(target_is_time=False, on_target=0, off_target=num_steps, tolerance=5, multi_spike=False)

    # calculate loss based on the network type and output decoding
    def loss(self, output, y_true):
        if self.spiking:
            y_pred = output[0][-1]
            
            if self.output_decoding=="rate":
                loss = self.loss_fn(y_pred, y_true)
            else:
                loss = self.loss_fn(y_pred, y_true)
        else:
            y_pred = output
            loss = self.loss_fn(y_pred, y_true) 

        return loss

    # return the predicted class from spike trains
    def return_predicted(self, output):
        if not self.spiking:
            ps = torch.exp(output)
            top_class = torch.argmax(ps, dim=1)
            
            return top_class
        else:
            y_pred = output[0][-1]
            _, predicted = y_pred.sum(dim=0).max(dim=1)

            return predicted
        
    # forward pass based on the network type
    def forward_pass(self, net, x, num_steps, gaussian=False, time_first=False, real_time=False):
        if self.net_type=="OC_SCNN":
            if gaussian:
                return net(x, num_steps, gaussian=True)
            
            return net(x, num_steps)
        elif self.net_type=="CSNN":
            return net(x, num_steps)
        elif self.net_type=="SNN":
            return net(x, num_steps, time_first=time_first, real_time=real_time)
        elif self.net_type=="CNN" or self.net_type=="RNN":
            return net(x)
    
    # calculate the confusion matrix
    def perf_measure(self, y_actual, y_hat):

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                self.TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                self.FP += 1
            if y_actual[i]==y_hat[i]==0:
                self.TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                self.FN += 1
        
    # calculate the precision
    def precision(self):
        return self.TP/(self.TP+self.FP+1e-8)

    # calculate the recall
    def recall(self):
        return self.TP/(self.TP+self.FN+1e-8)
    
    # calculate the harmonic mean of precision and recall
    def f1_score(self):
        return 2*(self.precision()*self.recall())/(self.precision()+self.recall()+1e-8)

    # calculate the accuracy
    def accuracy(self, output, y_true):
        if not self.spiking:
            ps = torch.exp(output)
            top_class = torch.argmax(ps, dim=1)
            equals = top_class == y_true

            return torch.mean(equals.type(torch.FloatTensor)).item()
        else:
            y_pred = output[0][-1]
            if self.output_decoding=="rate":
                return functional.acc.accuracy_rate(y_pred,y_true)
            else:
                return functional.acc.accuracy_temporal(y_pred,y_true)
        