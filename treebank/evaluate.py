from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from collections import defaultdict
import glob
import pickle
import time, datetime
import os


class EvaluateBoundary(object):
    """
    Metrics: precision/recall, F1
    Ground-truth: Penn Treebank
    """
    def __init__(self, file_truth, file_layers_predict, 
                 loss_file='../hierarchical-rnn/loss.tmp', 
                 pickle_path='.'):
        self.file_truth = file_truth
        self.file_layers_predict = file_layers_predict
        self.loss_file = loss_file
        self.pickle_path=pickle_path
        self._get_labels()

    def _get_labels(self):
        labelize = lambda bounds: np.array(list(bounds.strip()))

        with open(self.file_truth, 'r') as f:
            self.truth = labelize(f.read())

        self.pred_layers = defaultdict()
        for f in glob.glob(self.file_layers_predict):
            with open(f, 'r') as file:
                self.pred_layers[f] = labelize(file.read())
                if len(self.pred_layers[f]) > len(self.truth):
                    raise Exception("More predicted points than truth.")

        return self.truth, self.pred_layers

    def evaluate(self, average=None, read_loss=True):
        """
        Evaluate predictions for each layer of (precision, recall, f1, support)
        :param average: same as http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        :return: a dictionary with layer as key, (precision, recall, f1, support) as value
                (each element contains two values for (1, 0))
        """
        self.prec_recall_f1 = defaultdict(tuple)
        for l, layer_pred in self.pred_layers.items():
            precision, recall, f1, support = precision_recall_fscore_support(
                self.truth[:len(self.pred_layers[l])], self.pred_layers[l], average=average)
            self.prec_recall_f1[l] = (precision, recall, f1, support)

        if read_loss:
            self._read_loss()
        return self.prec_recall_f1

    def _read_loss(self):
        try:
            with open(self.loss_file, 'r') as f:
                loss = float(f.read())
        except:
            raise
        os.remove(self.loss_file)
        self.prec_recall_f1["bpc"] = loss

    def save_eval(self, name=None):
        print(self.prec_recall_f1)
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%H%M%S')
        if name == None:        
            pickle.dump(self.prec_recall_f1, open(self.pickle_path + "/eval_{}.pkl".format(timestamp), 'wb'))
        else:
            pickle.dump(self.prec_recall_f1, open(self.pickle_path + "/" + name, 'wb'))

if __name__ == '__main__':
    eval_label = EvaluateBoundary("corpora/boundaries.txt", "layer_*.txt")
    eval_label.evaluate()
    eval_label.save_eval()
