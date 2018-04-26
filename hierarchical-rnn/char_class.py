from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, viz_char_boundaries
from inventory_lm import *

batches_in, batches_out = prepare_inputs(batch_size=BATCH_SIZE,
                                         num_batches=NUM_BATCHES,
                                         truncate_len=TRUNCATE_LEN,
                                         step_size=STEP_SIZE,
                                         text_path='text8.txt')

network = HMLSTMNetwork(output_size=OUTPUT_SIZE, input_size=INPUT_SIZE, embed_size=EMBED_SIZE,
                        out_hidden_size=OUT_HIDDEN_SIZE, hidden_state_sizes=HIDDEN_STATE_SIZES,
                        task='classification')

network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path='./text8', epochs=EPOCHS)

predictions = network.predict(batches_in[-1], variable_path='./text8')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./text8')

# visualize boundaries
viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])
