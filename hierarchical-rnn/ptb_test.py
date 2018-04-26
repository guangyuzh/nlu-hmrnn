from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, save_boundaries
import os, glob
from inventory_lm import *


def _rm_obsolete_pred(path):
    for f in glob.glob(path + "*.txt"):
        os.remove(f)


batches_in, batches_out = prepare_inputs(batch_size=1,
                                         truncate_len=TRUNCATE_LEN, step_size=TRUNCATE_LEN,
                                         text_path='../treebank/corpora/sentences.txt')

network = HMLSTMNetwork(output_size=OUTPUT_SIZE, input_size=INPUT_SIZE, embed_size=EMBED_SIZE,
                        out_hidden_size=OUT_HIDDEN_SIZE, hidden_state_sizes=HIDDEN_STATE_SIZES,
                        task='classification')

path = "../treebank/"
# Clear previous prediction outputs
_rm_obsolete_pred(path)

tot_loss = 0
for b in batches_in:
    predictions, loss = network.predict(b, variable_path='./text8', return_loss=True)
    tot_loss += loss
    boundaries = network.predict_boundaries(b, variable_path='../treebank/corpora/sentences.txt')
    # visualize boundaries
    save_boundaries(get_text(b[0]), get_text(predictions[0]), boundaries[0], layers=[0, 1, 2], path=path)
tot_loss /= len(batches_in)

with open("loss.tmp", 'w') as f:
    f.write(str(tot_loss))
