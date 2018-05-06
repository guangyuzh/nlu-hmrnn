from hmlstm import get_text, save_boundaries
import os, glob
from configuration import *


def _rm_obsolete_pred(path):
    for f in glob.glob(path + "*.txt"):
        os.remove(f)


hparams = select_config()
batches_in, batches_out = hparams.pre_inputs('../treebank/corpora/sentences.txt', train=False)
network = hparams.gen_network()

path = "../treebank/"
# Clear previous prediction outputs
_rm_obsolete_pred(path)

# Compute mean loss for all batches
# Note: this may be an approximation towards the mean, because size(last_batch) may be smaller.
#       However, margin should be trivial for ample num_batches
tot_loss = 0
for b in batches_in:
    predictions, loss = network.predict(b, variable_path='./text8', return_loss=True)
    tot_loss += loss
    boundaries = network.predict_boundaries(b)
    # save layer-wise binary boundary indicators, predicted by the loaded model
    save_boundaries(get_text(b[0]), get_text(predictions[0]), boundaries[0],
                    layers=[i for i in range(hparams.num_layers)], path=path)
tot_loss /= len(batches_in)

with open("loss.tmp", 'w') as f:
    f.write(str(tot_loss))
