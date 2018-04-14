from hmlstm import HMLSTMNetworkQa, CBTDataset

VOCABULARY_PATH = './CBT_CN_vocab.txt'
TRAIN_DATA_PATH = './CBTest/data/cbtest_CN_quick_dev_6ex.txt'
INPUT_EMBED_SIZE = 128
CANDIDATE_NUM = 10

# prepare data pipeline
print("Preparing dataset...")
cbt = CBTDataset()
cbt.load_vocab(VOCABULARY_PATH)
train_dataset = cbt.prepare_dataset(TRAIN_DATA_PATH) # return a tf.data.Dataset instance

print("Preparing network...")
network = HMLSTMNetworkQa(output_size=CANDIDATE_NUM, input_size=INPUT_EMBED_SIZE, embed_size=INPUT_EMBED_SIZE, 
                        out_hidden_size=1024, hidden_state_sizes=1024, num_layers=2,
                        task='classification')

print("Training...")
network.train(cbt, train_dataset, save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path='./text8')


"""
predictions = network.predict(batches_in[-1], variable_path='./text8')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./text8')

# visualize boundaries
viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])
"""
