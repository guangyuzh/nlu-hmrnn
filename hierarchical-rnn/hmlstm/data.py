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

            if convert_word_to_id:
                signals['context'].append(self.words_to_ids(context))
                signals['query'].append(self.words_to_ids(query))
                signals['candidates'].append(np.array([self.word_to_id(word) for word in candidates], dtype=np.int32))
                signals['answer'].append(self.words_to_ids(answer))
            else:
                signals['context'].append(context)
                signals['query'].append(query)
                signals['candidates'].append(candidates)
                signals['answer'].append(answer)

        self.sample_num = len(signals['answer'])
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

    def prepare_dataset(self, text_path):
        """ return a tf.data.Dataset instance """
        signals = self.load_cbt(text_path)
        signals = self.merge_query_context(signals)
        dataset = tf.data.Dataset.from_tensor_slices((signals['query_context'], signals['answer'], signals['candidates']))
        dataset = dataset.shuffle(buffer_size=10000)
        # dataset = dataset.padded_batch(1, padded_shapes=[None])
        dataset = dataset.batch(self.batch_size)
        return dataset

    def convert_to_tensors(self, batch_data):
        """
        Convert a list of variable length string to a numpy array,
        converting words to ids automatically and padding may apply for batch_query_context
        batch_data: (batch_query_context, batch_answer, batch_candidates)
        """
        assert len(batch_data) == 3
        # deal with query_context
        qc_list = []
        max_length = 0 
        for query_context in batch_data[0]:
            qc_nparray = np.array([self.word_to_id(word) for word in query_context.decode('utf-8').split(' ')], dtype=np.int32)
            max_length = max(max_length, len(qc_nparray))
            qc_list.append(qc_nparray[None, :])     # [1, num_of_words]

        # add padding to query_context numpy array to make them same length
        pad_id = self.word_to_id(PAD)
        padded_qc_list = []
        for qc_nparray in qc_list:
            pad_size = max_length - qc_nparray.shape[1]
            padded_qc_nparray = np.append(qc_nparray, [[pad_id] * pad_size], axis=1) # [1, num_of_words + num_of_padding]
            padded_qc_list.append(padded_qc_nparray)

        converted_batch_query_context = np.concatenate(padded_qc_list, axis=0)

        # deal with answers
        converted_batch_answer = np.array([self.word_to_id(answer.decode('utf-8')) for answer in batch_data[1]], dtype=np.int32)

        # deal with candidates
        cand_list = []
        for candidates in batch_data[2]:
            cand_array = np.array([self.word_to_id(candidate.decode('utf-8')) for candidate in candidates], dtype=np.int32)
            cand_list.append(cand_array[None, :])
        converted_batch_candidates = np.concatenate(cand_list, axis=0)

        return converted_batch_query_context, converted_batch_answer, converted_batch_candidates

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

