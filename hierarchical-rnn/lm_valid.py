from hmlstm import get_text, viz_char_boundaries
from configuration import *


hparams = select_config()
batches_in, batches_out = hparams.pre_inputs('text8.txt')
valid_batches_in, valid_batches_out = hparams.pre_inputs('../treebank/corpora/sentences.txt', train=False)
network = hparams.gen_network()

network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path='./text8', epochs=hparams.epochs,
              valid_batches_in=valid_batches_in, valid_batches_out=valid_batches_out,
              valid_after_step=hparams.valid_after_step)
