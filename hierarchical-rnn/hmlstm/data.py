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

        self.word_id = {w: i for i, w in enumerate(text.split('\n'))}
        self.id_word = {i: w for w, i in self.word_id.items()}
        self.batch_size = batch_size
        self.sample_num = {}

    # def load_vocab(self, vocab_path):
    #     with open(vocab_path, 'r') as f:
    #         text = f.read()

    #     self.word_id = {w: i for i, w in enumerate(text.split('\n'))}
    #     self.id_word = {i: w for w, i in self.word_id.items()}
    #     return self.word_id, self.id_word

    def word_to_id(self, word):
        """ return a id correspondent to a word, or id to UNK if word not in vocabulary"""
        return self.word_id.get(word, self.word_id[UNK])

    def words_to_ids(self, s):
        """ return a id representation of a sentence"""
        return np.array([self.word_to_id(word) for word in s.split(' ')], dtype=np.int32)

    def id_to_word(self, id):
        return self.id_word.get(id, UNK)

    def ids_to_words(self, ids):
        # works only for 1-dim nparray
        return ' '.join([self.id_to_word(id) for id in ids])

    def load_cbt(self, text_path, convert_word_to_id=False):
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

            if convert_word_to_id:
                signals['context'].append(self.words_to_ids(context))
                signals['query'].append(self.words_to_ids(query))
                signals['candidates'].append(np.array([self.word_to_id(word) for word in candidates], dtype=np.int32))
            else:
                signals['context'].append(context)
                signals['query'].append(query)
                signals['candidates'].append(candidates)
            signals['answer'].append(ans_index) # store the index of answer instead of answer itself.
        return signals

    def merge_query_context(self, signals, separator=SEP, convert_word_to_id=False):
        signals['query_context'] = []
        for query, context in zip(signals['query'], signals['context']):
            if convert_word_to_id:
                query_context = np.concatenate((np.append(query, self.word_to_id(separator)), context), axis=0)
                signals['query_context'].append(query_context)
            else:
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
        vocab.add(UNK)
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
            for word in vocab:
                f.write(word + '\n')

