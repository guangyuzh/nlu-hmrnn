from hmlstm import get_text, viz_char_boundaries
from configuration import *


hparams = select_config()
batches_in, batches_out = hparams.pre_inputs('text8.txt')
ptb_batches_in, ptb_batches_out = hparams.pre_inputs('../treebank/corpora/sentences.txt', train=False)
network = hparams.gen_network()

network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path=hparams.output_dir + 'text8', epochs=hparams.epochs,
              val_batches_in=batches_in[-1:], val_batches_out=batches_out[-1:],
              val_after_step=hparams.valid_after_step)
