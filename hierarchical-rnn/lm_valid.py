from hmlstm import get_text, viz_char_boundaries
from configuration import *


hparams = select_config()
batches_in, batches_out = hparams.pre_inputs('text8.txt')
ptb_batches_in, ptb_batches_out = hparams.pre_inputs('../treebank/corpora/sentences.txt', train=False)
network = hparams.gen_network()

val_batch_num = 10
if hparams.quick_dev:
    val_batch_num = 1

network.train(batches_in[:-val_batch_num], batches_out[:-val_batch_num], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path=hparams.output_dir + 'text8', epochs=hparams.epochs,
              val_batches_in=batches_in[-val_batch_num:], val_batches_out=batches_out[-val_batch_num:],
              ptb_batches_in=ptb_batches_in, ptb_batches_out=ptb_batches_out,
              val_after_step=hparams.valid_after_step)
