from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, save_boundaries

batches_in, batches_out = prepare_inputs(batch_size=20, truncate_len=1000,
                                         step_size=500, text_path='../treebank/corpora/sentences.txt')

network = HMLSTMNetwork(output_size=27, input_size=27, embed_size=2048,
                        out_hidden_size=1024, hidden_state_sizes=1024,
                        task='classification')

for b in batches_in:
    predictions = network.predict(b, variable_path='./text8')
    boundaries = network.predict_boundaries(b, variable_path='../treebank/corpora/sentences.txt')
    # visualize boundaries
    save_boundaries(get_text(b[0]), get_text(predictions[0]), boundaries[0], layers=[0, 1, 2])
