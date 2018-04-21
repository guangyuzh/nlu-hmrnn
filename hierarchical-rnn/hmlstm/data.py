import tensorflow as tf
import numpy as np

UNK = '<UKN>'   # Unknown
SEP = '<SEP>'   # Separator
PAD = '<PAD>'   # Padding

"""
Usage:
cbt = CBTDataset(vocab_path, batch_size)
ds = cbt.prepare_dataset(train_text_path) # return a tf.data.Dataset instance
iterator = ds.make_one_shot_iterator()
query_context, candidates, answer = iterator.get_next()
...

"""

def load_cbt(text_path):
    """
    Read CBT dataset and return a list of tuples in the form of
    (context, query, candidates, answer).
    """
    with open(text_path, 'r') as f:
        text = f.read().lower()

    signals = {'context': [], 'query': [], 'candidates': [], 'answer': []}
    # parsing sample
    samples = text.split("\n\n")[:-1] # ignore the last '\n'
    for sample in samples:
        lines = sample.split("\n")
        assert len(lines) == 21

        context = ""
        for i in range(20):
            offset = (i + 1) // 10 + 2
            context += lines[i][offset:] + ' '

        # parsing query, candidates, answer
        query_ans_cand = lines[20].replace("\t\t", "\t").split("\t")
        assert len(query_ans_cand) == 3
        query = query_ans_cand[0][3:] # ignore the characters '21 ' in the begining
        answer = query_ans_cand[1]
        candidates = query_ans_cand[2].split('|')[:10]
        assert len(candidates) == 10
        ans_index = candidates.index(answer)

        signals['context'].append(context)
        signals['query'].append(query)
        signals['candidates'].append(candidates)
        signals['answer'].append(ans_index) # store the index of answer instead of answer itself.
    return signals

def merge_query_context(signals, separator=SEP):
    signals['query_context'] = []
    for query, context in zip(signals['query'], signals['context']):
        signals['query_context'].append(query + ' ' + separator + ' ' + context)
    del signals['query']
    del signals['context']
    return signals

def prepare_signals(text_path):
    # preprocess signals
    signals = load_cbt(text_path)
    signals = merge_query_context(signals)
    return signals

def create_vocabulary(input_path, output_path):
    """ Parse the file and create vocabulary"""
    with open(input_path, 'r') as f:
        text = f.read().lower()

    vocab = set()
    vocab.add(SEP)
    vocab.add(PAD)
    samples = text.split("\n\n")[:-1] # ignore the last '\n'
    for sample in samples:
        lines = sample.split("\n")
        assert len(lines) == 21

        context = ""
        for i in range(20):
            offset = (i + 1) // 10 + 2
            context += lines[i][offset:] + ' '

        # parsing query, candidates, answer
        query_ans_cand = lines[20].replace("\t\t", "\t").split("\t")
        assert len(query_ans_cand) == 3
        query = query_ans_cand[0][3:] # ignore the characters '21 ' in the begining
        answer = query_ans_cand[1]
        candidates = query_ans_cand[2].split('|')[:10]
        if len(candidates) != 10:
            raise ValueError('Sample has %d candidates, 10 required. Sample: %s' \
                % (len(candidates), sample))

        denoised_sample = context + ' ' + query + ' ' + answer + ' ' + ' '.join(candidates)
        for word in denoised_sample.split(' '):
            vocab.add(word)

    # save vocabulary to file
    print("Vocab size for %s: %d" % (input_path, len(vocab)))
    with open(output_path, 'w') as f:
        f.write('\n'.join(vocab))

