from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, save_boundaries
import os, glob
from configuration import *


def _rm_obsolete_pred(path):
    for f in glob.glob(path + "*.txt"):
        os.remove(f)


batches_in, batches_out = prepare_inputs(batch_size=hparams.batch_size,
                                         truncate_len=hparams.truncate_len,
                                         step_size=hparams.step_size,
                                         text_path='../treebank/corpora/sentences.txt')

path = "../treebank/"
# Clear previous prediction outputs
_rm_obsolete_pred(path)

tot_loss = 0
for b in batches_in:
    predictions, loss = network.predict(b, variable_path='./text8', return_loss=True)
    tot_loss += loss
    boundaries = network.predict_boundaries(b, variable_path='../treebank/corpora/sentences.txt')
    # visualize boundaries
    save_boundaries(get_text(b[0]), get_text(predictions[0]), boundaries[0],
                    layers=[i for i in range(hparams.num_layers)], path=path)
tot_loss /= len(batches_in)

with open("loss.tmp", 'w') as f:
    f.write(str(tot_loss))
