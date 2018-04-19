from hmlstm import HMLSTMNetworkQa, CBTDataset
import argparse

VOCABULARY_PATH = './CBTest/vocab/CBT_CN_vocab.txt'
TRAIN_DATA_PATH = './CBTest/data/cbtest_CN_train.txt'
VALID_DATA_PATH = './CBTest/data/cbtest_CN_valid_2000ex.txt'
TEST_DATA_PATH = './CBTest/data/cbtest_CN_test_2500ex.txt'
INPUT_EMBED_SIZE = 128
CANDIDATE_NUM = 10
BATCH_SIZE = 32

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dev", help="help catch bug early by running a subset of data",
					action="store_true")
parser.add_argument("--predict", help="use trained network to predict",
					action="store_true")
args = parser.parse_args()

if args.dev:
	TRAIN_DATA_PATH = './CBTest/data/cbtest_CN_quick_dev_6ex.txt'
	BATCH_SIZE = 2

print("Preparing dataset...")
cbt = CBTDataset(vocab_path=VOCABULARY_PATH, batch_size=BATCH_SIZE)
train_dataset = cbt.prepare_dataset(TRAIN_DATA_PATH) # return a tf.data.Dataset instance

print("Preparing network...")
network = HMLSTMNetworkQa(output_size=CANDIDATE_NUM, input_size=INPUT_EMBED_SIZE, embed_size=INPUT_EMBED_SIZE, 
                        out_hidden_size=1024, hidden_state_sizes=1024, num_layers=2,
                        task='classification')

if not args.predict:
	print("Training...")
	network.train(cbt, train_dataset, save_vars_to_disk=True, 
	              load_vars_from_disk=False, variable_path='./qa_variable')
else:
	print("Predicting...")

"""
predictions = network.predict(batches_in[-1], variable_path='./text8')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./text8')

# visualize boundaries
viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])
"""
