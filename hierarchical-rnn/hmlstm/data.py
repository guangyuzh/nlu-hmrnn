import tensorflow as tf
import numpy as np

UNK = '<UKN>'   # Unknown
SEP = '<SEP>'   # Separator
PAD = '<PAD>'   # Padding

class CBTDataset(object):
    """
    Usage:
    cbt = CBTDataset(vocab_path, batch_size)
    cbt.load_vocab(vocab_path)
    ds = cbt.prepare_dataset(text_path) # return a tf.data.Dataset instance

    use convert_to_tensors() to do the padding and convert words to ids
        on what returns after calling iterator.get_next()
    """
    def __init__(self, vocab_path, batch_size=2):
        with open(vocab_path, 'r') as f:
            text = f.read()

        vocab = tf.constant(text.split('\n'))
        self.table = tf.contrib.lookup.index_table_from_tensor(mapping=vocab)
        self.reverse_table = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=vocab, default_value=UNK)

        self.batch_size = batch_size
        self.sample_num = {}

    def lookup(self, data):
        """
        Convert word strings to word ids
        """
        return self.table.lookup(data)

    def reverse_lookup(self, data):
        """
        Convert indexes to words
        """
        return self.reverse_table.lookup(data)

    def load_cbt(self, text_path):
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

    def merge_query_context(self, signals, separator=SEP):
        signals['query_context'] = []
        for query, context in zip(signals['query'], signals['context']):
            signals['query_context'].append(query + ' ' + separator + ' ' + context)
        del signals['query']
        del signals['context']
        return signals

    def prepare_dataset(self, text_path, name='train'):
        """ return a tf.data.Dataset instance """
        signals = self.load_cbt(text_path)
        signals = self.merge_query_context(signals)
        self.sample_num[name] = len(signals['answer'])
        qc_ds = tf.data.Dataset.from_tensor_slices(signals['query_context'])
        qc_ds = qc_ds.map(lambda str: tf.string_split([str]).values)
        qc_ds = qc_ds.map(lambda word: self.lookup(word))

        cand_ds = tf.data.Dataset.from_tensor_slices(signals['candidates'])
        cand_ds = cand_ds.map(lambda cand: self.lookup(cand))

        ans_ds = tf.data.Dataset.from_tensor_slices(signals['answer'])

        ds = tf.data.Dataset.zip((qc_ds, cand_ds, ans_ds))
        ds = ds.shuffle(buffer_size=10000)
        pad_index = self.lookup(tf.constant('<PAD>'))
        ds = ds.padded_batch(batch_size=self.batch_size,
                padded_shapes=([None], [10], []),               # 2nd and 3rd component (cand & ans) not padded
                padding_values=(pad_index, np.int64(-1), 0))    # 2nd and 3rd component (cand & ans) not padded
        return ds

    @staticmethod
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

