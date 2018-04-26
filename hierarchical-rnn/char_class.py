from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, viz_char_boundaries
from configuration import *


batches_in, batches_out = prepare_inputs(batch_size=hparams.batch_size,
                                         num_batches=hparams.num_batches,
                                         truncate_len=hparams.truncate_len,
                                         step_size=hparams.step_size,
                                         text_path='text8.txt')

network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path='./text8', epochs=hparams.epochs)

predictions = network.predict(batches_in[-1], variable_path='./text8')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./text8')

# visualize boundaries
viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])
