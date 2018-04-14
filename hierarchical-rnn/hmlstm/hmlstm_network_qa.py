from .hmlstm_cell import HMLSTMCell, HMLSTMState
from .multi_hmlstm_cell import MultiHMLSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import numpy as np
import sys


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

CAND_NUM = 10 # candidates number

class HMLSTMNetworkQa(object):
    def __init__(self,
                 input_size=1,
                 output_size=CAND_NUM,
                 num_layers=3,
                 hidden_state_sizes=50,
                 out_hidden_size=100,
                 embed_size=100,
                 vocab_size=55000,
                 task='regression'):
        """
        HMLSTMNetworkQa is a class representing hierarchical multiscale
        long short-term memory network modified specifically for task 
        Reading Comprehension.

        params:
        ---
        input_size: integer, the size of an input at one timestep
        output_size: integer, the size of an output at one timestep
        num_layers: integer, the number of layers in the hmlstm
        hidden_state_size: integer or list of integers. If it is an integer,
            it is the size of the hidden state for each layer of the hmlstm.
            If it is a list, it must have length equal to the number of layers,
            and each integer of the list is the size of the hidden state for
            the layer correspodning to its index.
        out_hidden_size: integer, the size of the two hidden layers in the
            output network.
        embed_size: integer, the size of the embedding in the output network.
        task: string, one of 'regression' and 'classification'.
        """

        self._out_hidden_size = out_hidden_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._input_size = input_size
        self._session = None
        self._graph = None
        self._task = task
        self._output_size = output_size
        self._vocab_size = vocab_size

        if type(hidden_state_sizes) is list \
            and len(hidden_state_sizes) != num_layers:
            raise ValueError('The number of hidden states provided must be the'
                             + ' same as the nubmer of layers.')

        if type(hidden_state_sizes) == int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        if task == 'classification':
            self._loss_function = tf.nn.softmax_cross_entropy_with_logits
        elif task == 'regression':
            self._loss_function = lambda logits, labels: tf.square((logits - labels))

        # batch_in_shape = (None, None, self._input_size)
        # batch_out_shape = (None, None, self._output_size)
        # self.batch_in = tf.placeholder(
        #     tf.float32, shape=batch_in_shape, name='batch_in')


        batch_qc_shape = (None, None)                # [num_of_timesteps, Batch_size]
        batch_cand_shape = (None, self._output_size) # [Batch_size, output_size]
        batch_out_shape = (None, self._output_size)  # [Batch_size, output_size]

        self.batch_qc = tf.placeholder(
            tf.int32, shape=batch_qc_shape, name='batch_qc')
        self.batch_cand = tf.placeholder(
            tf.int32, shape=batch_cand_shape, name='batch_cand')
        self.batch_out = tf.placeholder(
            tf.float32, shape=batch_out_shape, name='batch_out')

        self._optimizer = tf.train.AdamOptimizer(1e-3)

        self._initialize_input_module_variables()
        self._initialize_output_variables()
        self._initialize_gate_variables()
        self._initialize_embedding_variables()

    def _initialize_input_module_variables(self):
        with vs.variable_scope('input_module_vars'):
            embed_table = vs.get_variable(
                'embed_table', shape=[self._vocab_size, self._embed_size])

    def _initialize_gate_variables(self):
        with vs.variable_scope('gates_vars'):
            for l in range(self._num_layers):
                vs.get_variable(
                    'gate_%s' % l, [sum(self._hidden_state_sizes), 1],
                    dtype=tf.float32)

    def _initialize_embedding_variables(self):
        with vs.variable_scope('embedding_vars'):
            embed_shape = [sum(self._hidden_state_sizes), self._embed_size]
            vs.get_variable('embed_weights', embed_shape, dtype=tf.float32)

    def _initialize_output_variables(self):
        with vs.variable_scope('output_module_vars'):
            vs.get_variable('b1', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b2', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b3', [1, self._output_size], dtype=tf.float32)
            vs.get_variable(
                'w1', [self._embed_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w2', [self._out_hidden_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w3', [self._out_hidden_size, self._output_size],
                dtype=tf.float32)

    def load_variables(self, path='./hmlstm_ckpt'):
        if self._session is None:
            self._session = tf.Session()

            saver = tf.train.Saver()
            print('loading variables...')
            saver.restore(self._session, path)

    def save_variables(self, path='./hmlstm_ckpt'):
        saver = tf.train.Saver()
        print('saving variables...')
        saver.save(self._session, path)

    def input_module(self, batch_text):
        """
        Convert word id to word embedding.

        batch_text: [B, #words], dtype=int32 (texts represented in the form of word ids)
        return: [B, #words, embedding], dtype=float32
        """
        with vs.variable_scope('input_module_vars', reuse=True):
            embed_table = vs.get_variable('embed_table')
            # input module embedding size the same as output module embedding size
        return tf.nn.embedding_lookup(embed_table, batch_text)

    def gate_input(self, hidden_states):
        '''
        gate the incoming hidden states
        hidden_states: [B, sum(h_l)]

        gated_input: [B, sum(h_l)]
        '''
        with vs.variable_scope('gates_vars', reuse=True):
            gates = []  # [[B, 1] for l in range(L)]
            for l in range(self._num_layers):
                weights = vs.get_variable('gate_%d' % l, dtype=tf.float32)
                gates.append(tf.sigmoid(tf.matmul(hidden_states, weights)))

            split = array_ops.split(
                value=hidden_states,
                num_or_size_splits=self._hidden_state_sizes,
                axis=1)

            gated_list = []  # [[B, h_l] for l in range(L)]
            for gate, hidden_state in zip(gates, split):
                gated_list.append(tf.multiply(gate, hidden_state))

            gated_input = tf.concat(gated_list, axis=1)  # [B, sum(h_l)]
        return gated_input

    def embed_input(self, gated_input):
        '''
        gated_input: [B, sum(h_l)]

        embedding: [B, E], i.e. [B, embed_size]
        '''
        with vs.variable_scope('embedding_vars', reuse=True):
            embed_weights = vs.get_variable('embed_weights', dtype=tf.float32)

            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(prod)

        return embedding

    def output_module(self, embedding, outcome):
        '''
        embedding: [B, E]
        outcome: [B, output_size]

        loss: [B, output_size] or [B, 1]
        prediction: [B, output_size]
        '''
        with vs.variable_scope('output_module_vars', reuse=True):
            b1 = vs.get_variable('b1')
            b2 = vs.get_variable('b2')
            b3 = vs.get_variable('b3')
            w1 = vs.get_variable('w1')
            w2 = vs.get_variable('w2')
            w3 = vs.get_variable('w3')

            # feed forward network
            # first layer
            l1 = tf.nn.tanh(tf.matmul(embedding, w1) + b1)

            # second layer
            l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            # the loss function used below
            # softmax_cross_entropy_with_logits
            prediction = tf.add(tf.matmul(l2, w3), b3, name='prediction')

            loss_args = {'logits': prediction, 'labels': outcome}
            loss = self._loss_function(**loss_args)

            if self._task == 'classification':
                # due to nature of classification loss function
                loss = tf.expand_dims(loss, -1)

        return loss, prediction

    def output_module_qa(self, embedding, batch_cand_embed, outcome):
        '''
        embedding:          [B, E]                  query_context embedding
        batch_cand_embed:   [B, output_size, E]
        outcome:            [B, output_size]

        loss: [B, output_size] or [B, 1]
        logits: [B, output_size]
        '''
        embedding = tf.expand_dims(embedding, axis=-1)              # [B, E, 1]
        product = tf.matmul(batch_cand_embed, embedding)            # [B, output_size, 1]
        logits = tf.squeeze(product, axis=2, name='logits')         # [B, output_size]

        # the loss function used below
        # softmax_cross_entropy_with_logits
        loss_args = {'logits': logits, 'labels': outcome}
        loss = self._loss_function(**loss_args)

        if self._task == 'classification':
            # due to nature of classification loss function
            loss = tf.expand_dims(loss, -1)                         # [B, 1]

        return loss, logits

    def get_ans_prob_distribution(self, batch_cand, batch_ans):
        """
        Convert batch_ans to probablistic distribution,
        i.e., if ans='dog' and candidates=['cat', 'dog','turtle', 'rhino'],
        then return out=[0, 1, 0, 0]

        batch_cand: [B, output_size], words represented by word id
        batch_ans:  [B], words represented by word id

        return batch_out: [B, output_size]
        """
        batch_size = batch_cand.shape[0]
        batch_out = np.zeros([batch_size, self._output_size])
        for i in range(batch_size):
            for j in range(self._output_size):
                if batch_cand[i][j] == batch_ans[i]:
                    batch_out[i][j] = 1.0
                    break

        return batch_out

    def create_multicell(self, batch_size, reuse):
        def hmlstm_cell(layer):
            if layer == 0:
                h_below_size = self._input_size
            else:
                h_below_size = self._hidden_state_sizes[layer - 1]

            if layer == self._num_layers - 1:
                # doesn't matter, all zeros, but for convenience with summing
                # so the sum of ha sizes is just sum of hidden states
                h_above_size = self._hidden_state_sizes[0]
            else:
                h_above_size = self._hidden_state_sizes[layer + 1]

            return HMLSTMCell(self._hidden_state_sizes[layer], batch_size,
                              h_below_size, h_above_size, reuse)

        hmlstm = MultiHMLSTMCell(
            [hmlstm_cell(l) for l in range(self._num_layers)], reuse)

        return hmlstm

    def split_out_cell_states(self, accum):
        '''
        accum: [B, H], i.e. [B, sum(h_l) * 2 + num_layers]


        cell_states: a list of ([B, h_l], [B, h_l], [B, 1]), with length L
        '''
        splits = []
        for size in self._hidden_state_sizes:
            splits += [size, size, 1]

        split_states = array_ops.split(value=accum,
                                       num_or_size_splits=splits, axis=1)

        cell_states = []
        for l in range(self._num_layers):
            c = split_states[(l * 3)]
            h = split_states[(l * 3) + 1]
            z = split_states[(l * 3) + 2]
            cell_states.append(HMLSTMState(c=c, h=h, z=z))

        return cell_states

    def get_h_aboves(self, hidden_states, batch_size, hmlstm):
        '''
        hidden_states: [[B, h_l] for l in range(L)]

        h_aboves: [B, sum(ha_l)], ha denotes h_above
        '''
        concated_hs = array_ops.concat(hidden_states[1:], axis=1)

        h_above_for_last_layer = tf.zeros(
            [batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32)

        h_aboves = array_ops.concat(
            [concated_hs, h_above_for_last_layer], axis=1)

        return h_aboves

    def network(self, reuse):
        batch_in = self.input_module(self.batch_qc)                                 # [#timestep, B, E]
        batch_size = tf.shape(batch_in)[1]
        hmlstm = self.create_multicell(batch_size, reuse)

        def scan_rnn(accum, elem):
            # each element is the set of all hidden states from the previous
            # time step
            cell_states = self.split_out_cell_states(accum)

            h_aboves = self.get_h_aboves([cs.h for cs in cell_states],
                                         batch_size, hmlstm)    # [B, sum(ha_l)]
            # [B, I] + [B, sum(ha_l)] -> [B, I + sum(ha_l)]
            hmlstm_in = array_ops.concat((elem, h_aboves), axis=1)
            _, state = hmlstm(hmlstm_in, cell_states)
            # a list of (c=[B, h_l], h=[B, h_l], z=[B, 1]) ->
            # a list of [B, h_l + h_l + 1]
            concated_states = [array_ops.concat(tuple(s), axis=1) for s in state]
            return array_ops.concat(concated_states, axis=1)    # [B, H]
        # denote 'elem_len' as 'H'
        elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers
        initial = tf.zeros([batch_size, elem_len])              # [B, H]

        states = tf.scan(scan_rnn, batch_in, initial)      # [T, B, H]

        def map_indicators(elem):
            state = self.split_out_cell_states(elem)
            return tf.concat([l.z for l in state], axis=1)

        raw_indicators = tf.map_fn(map_indicators, states)      # [T, B, L]
        indicators = tf.transpose(raw_indicators, [1, 2, 0])    # [B, L, T]
        # to_map = tf.concat((states, self.batch_out), axis=2)    # [T, B, H + O]

        def map_output(elem, outcome):
            """
            Apply output module on one timestep

            elem: state at some timestep, [B, H]
            outcome: [B, output_size]
            return (loss, logits), [B, dim(loss) + dim(logits)]
            """
            hs = [s.h for s in self.split_out_cell_states(elem)]
            gated = self.gate_input(tf.concat(hs, axis=1))          # [B, sum(h_l)]
            embeded = self.embed_input(gated)                       # [B, E]
            batch_cand_embed = self.input_module(self.batch_cand)   # [B, output_size, E]
            loss, logits = self.output_module_qa(embeded, batch_cand_embed, outcome)
            # [B, output_size * 2] or [B, 1 + output_size]
            return tf.concat((loss, logits), axis=1)

        # mapped = tf.map_fn(map_output, to_map)                  # [T, B, _]
        # for QA, only need to apply map_output to the last timestep
        mapped = map_output(states[-1], self.batch_out)     # [B, dim(loss)+dim(pred)]

        # mapped has diffenent shape for task 'regression' and 'classification'
        loss = tf.reduce_mean(mapped[:, :-self._output_size])  # scalar
        predictions = mapped[:, -self._output_size:]
        train = self._optimizer.minimize(loss)

        return train, loss, indicators, predictions

    def train(self,
              cbt,
              dataset,
              variable_path='./hmlstm_ckpt',
              load_vars_from_disk=False,
              save_vars_to_disk=False,
              epochs=3):
        """
        Train the network.

        params:
        ---
        cbt: a CBTDataset instance
        dataset: a tf.data.Dataset instance that can return training data
        variable_path: the path to which variable values will be saved and/or
            loaded
        load_vars_from_disk: bool, whether to load variables prior to training
        load_vars_from_disk: bool, whether to save variables after training
        epochs: integer, number of epochs
        """

        optim, loss, _, _ = self._get_graph()

        if not load_vars_from_disk:
            if self._session is None:

                self._session = tf.Session()
                init = tf.global_variables_initializer()
                self._session.run(init)
        else:
            self.load_variables(variable_path)


        iterator = dataset.make_initializable_iterator()
        next_sample = iterator.get_next()
        for epoch in range(epochs):
            print('Epoch %d' % epoch)
            self._session.run(iterator.initializer)
            while True:
                try:
                    sample = self._session.run(next_sample)
                    batch_qc, batch_ans, batch_cand = cbt.convert_to_tensors(sample)
                    batch_out = self.get_ans_prob_distribution(batch_cand, batch_ans)
                    ops = [optim, loss]
                    feed_dict = {
                        self.batch_qc:   np.swapaxes(batch_qc, 0, 1),   # [T, B]
                        self.batch_cand: batch_cand,                    # [B, output_size]
                        self.batch_out:  batch_out,                     # [B, output_size]
                    }
                    _, _loss = self._session.run(ops, feed_dict)
                    print('loss:', _loss)
                except tf.errors.OutOfRangeError: 
                    # end of one epoch
                    break

        self.save_variables(variable_path)

    def predict(self, cbt, dataset, variable_path='./hmlstm_ckpt'):
        """
        Make predictions.

        params:
        ---
        cbt: a CBTDataset instance
        dataset: a tf.data.Dataset instance that can return training data
        # batch: batch for which to make predictions. should have dimensions
        #     [batch_size, num_timesteps, output_size]
        variable_path: string. If there is no active session in the network
            object (i.e. it has not yet been used to train or predict, or the
            tensorflow session has been manually closed), variables will be
            loaded from the provided path. Otherwise variables already present
            in the session will be used.

        returns:
        ---
        predictions for the batch
        """
        labels = tf.placeholder(tf.int32, shape=(None), name='labels')
        _, _, _, predictions = self._get_graph()
        self._load_vars(variable_path)

        # prepare dataset pipeline
        iterator = dataset.make_initializable_iterator()
        next_sample = iterator.get_next()
        self._session.run(iterator.initializer)

        # prepare metric
        accuracy, update_op = tf.metrics.accuracy(labels, predictions)
        sess.run(tf.local_variable_initializer())

        while True:
            try:
                sample = self._session.run(next_sample)
                batch_qc, batch_ans, batch_cand = cbt.convert_to_tensors(sample)
                batch_out = self.get_ans_prob_distribution(batch_cand, batch_ans)
                feed_dict = {
                    self.batch_qc:   np.swapaxes(batch_qc, 0, 1),   # [T, B]
                    self.batch_cand: batch_cand,                    # [B, output_size]
                    self.batch_out:  np.zeros(batch_out.shape),      # [B, output_size]
                    labels:          batch_out,
                }
                self._session.run(update_op, feed_dict)    # [B, output_size]

            except tf.errors.OutOfRangeError: 
                print("Valid Accuracy:", self._session.run(accuracy))                
                break


    def predict_boundaries(self, batch, variable_path='./hmlstm_ckpt'):
        """
        Find indicator values for every layer at every timestep.

        params:
        ---
        batch: batch for which to make predictions. should have dimensions
            [batch_size, num_timesteps, output_size]
        variable_path: string. If there is no active session in the network
            object (i.e. it has not yet been used to train or predict, or the
            tensorflow session has been manually closed), variables will be
            loaded from the provided path. Otherwise variables already present
            in the session will be used.

        returns:
        ---
        indicator values for ever layer at every timestep
        """

        batch = np.array(batch)
        _, _, indicators, _ = self._get_graph()

        self._load_vars(variable_path)

        # batch_out is not used for prediction, but needs to be fed in
        batch_out_size = (batch.shape[1], batch.shape[0], self._output_size)
        _indicators = self._session.run(indicators, {
            self.batch_in: np.swapaxes(batch, 0, 1),
            self.batch_out: np.zeros(batch_out_size)
        })

        return np.array(_indicators)

    def _get_graph(self):
        if self._graph is None:
            self._graph = self.network(reuse=False)
        return self._graph

    def _load_vars(self, variable_path):
        if self._session is None:
            try:
                self.load_variables(variable_path)
            except:
                raise RuntimeError('Session unitialized and no variables saved'
                                   + ' at provided path %s' % variable_path)
