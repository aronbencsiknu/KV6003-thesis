from snntorch import functional
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 

class Metrics():
    def __init__(self, net_type, set_type, output_decoding, train_set, device, num_steps):
        self.net_type = net_type
        self.set_type = set_type
        self.output_decoding = output_decoding
        self.train_set = train_set

        self.spiking = True if self.net_type=="CSNN" or self.net_type=="SNN" else False
        
        if self.set_type=="non-spiking":
            self.loss_fn = torch.nn.NLLLoss(train_set.weights4balance().to(device))
        elif self.output_decoding=="rate":
            #loss_fn = mse_count_loss(correct_rate=1, incorrect_rate=0.1)
            self.loss_fn = functional.loss.mse_count_loss(correct_rate=1, incorrect_rate=0.1)
        elif self.output_decoding=="latency":
            self.loss_fn = functional.loss.mse_temporal_loss(target_is_time=False, on_target=0, off_target=num_steps, tolerance=5, multi_spike=False)

    def loss(self, output, y_true):
        if self.spiking:
            y_pred = output[0][-1]
        else:
            y_pred = output  
        return self.loss_fn(y_pred, y_true)

    def accuracy(self, output, y_true):
        if not self.spiking:
            ps = torch.exp(output)
            top_class = torch.argmax(ps, dim=1)
            equals = top_class == y_true

            return torch.mean(equals.type(torch.FloatTensor)).item()
        else:
            y_pred = output[0][-1]
            #print(output)
            if self.output_decoding=="rate":
                return functional.acc.accuracy_rate(y_pred,y_true)
            else:
                return functional.acc.accuracy_temporal(y_pred,y_true)

    def return_predicted(self, output):
        if not self.spiking:
            ps = torch.exp(output)
            top_class = torch.argmax(ps, dim=1)
            return top_class
        else:
            y_pred = output[0][-1]
            _, predicted = y_pred.sum(dim=0).max(dim=1)

            return predicted
        
    def forward_pass(self, net, x, num_steps):
        if self.net_type=="CSNN":
            return net(x, num_steps)
        elif self.net_type=="SNN":
            return net(x, num_steps)
        elif self.net_type=="CNN":
            return net(x)
    """def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def f1_matrix(self):
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

    def roc_auc(self):
        return roc_auc_score(self.y_true, self.y_pred)

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def classification_report(self):
        return classification_report(self.y_true, self.y_pred)

    def all(self):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "roc_auc": self.roc_auc(),
            "confusion_matrix": self.confusion_matrix(),
            "classification_report": self.classification_report()
        }"""