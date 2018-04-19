import matplotlib.pyplot as plt
import sys
from collections import defaultdict

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


def plot_indicators(truth, prediction, indicators):
    f, ax = plt.subplots()
    ax.plot(truth, label='truth')
    ax.plot(prediction, label='prediction')
    ax.legend()

    colors = ['r', 'b', 'g', 'o', 'm', 'l', 'c']
    for l, layer in enumerate(indicators):
        for i, indicator in enumerate(layer):
            if indicator == 1.:
                p = 1.0 / len(indicators)
                ymin = p * l
                ymax = p * (l + 1)
                ax.axvline(i, color=colors[l], ymin=ymin, ymax=ymax, alpha=.3)

    return f


def viz_char_boundaries(truth, predictions, indicators, row_len=60):
    start = 0
    end = row_len
    while start < len(truth):
        for l in reversed(indicators):
            print(''.join([str(int(b)) for b in l])[start:end])
        print(predictions[start:end])
        print(truth[start:end])
        print()

        start = end
        end += row_len


def save_boundaries(truth, predictions, indicators, layers,
                    row_len=60, path="../treebank/"):
    """
    save predicted boundaries
    :param truth: one batch of input sentence
    :param predictions: predicted character sequence
    :param indicators: multi-layer boundary indicators
    :param layers: list; selected layer to save its indicators
    :param row_len: viz length
    :param path: save path
    :return: None
    """
    start = 0
    end = row_len
    layer_bounds = defaultdict(str)
    verbose = ""
    print("Start saving predicted boundaries")
    while start < len(truth):
        for i, l in enumerate(reversed(indicators)):
            if i in layers:
                layer_bounds[i] += ''.join([str(int(b)) for b in l])[start:end]
            verbose += ''.join([str(int(b)) for b in l])[start:end] + '\n'
        verbose += predictions[start:end] + '\n'
        verbose += truth[start:end] + '\n\n'
        start = end
        end += row_len

    for l, v in layer_bounds.items():
        with open(path + "layer_{}_bound.txt".format(l), 'w') as f:
            f.write(v)
            f.close()
    with open(path + "compare.txt", 'w') as f:
        f.write(verbose)
        f.close()
    print("Finished saving one batch")
