from hmlstm import get_text, viz_char_boundaries
from configuration import *


hparams = select_config()
batches_in, batches_out = hparams.pre_inputs('../treebank/corpora/sentences.txt', train=False)
network = hparams.gen_network()

for b_in, b_out in zip(batches_in, batches_out):
    predictions = network.predict(b_in, variable_path='./text8')
    boundaries = network.predict_boundaries(b_out, variable_path='./text8')
    # visualize boundaries
    viz_char_boundaries(get_text(b_out[0]), get_text(predictions[0]), boundaries[0])
