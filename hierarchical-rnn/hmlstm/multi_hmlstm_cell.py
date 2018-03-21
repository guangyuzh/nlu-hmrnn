from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf


class MultiHMLSTMCell(rnn_cell_impl.RNNCell):
    """HMLSTM cell composed squentially of individual HMLSTM cells."""

    def __init__(self, cells, reuse):
        super(MultiHMLSTMCell, self).__init__(_reuse=reuse)
        self._cells = cells

    def zero_state(self, batch_size, dtype):
        return [cell.zero_state(batch_size, dtype) for cell in self._cells]

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state.
        inputs: [B, I + sum(ha_l)]
        state: a list of ([B, h_l], [B, h_l], [B, 1]), with length L

        hidden_states: a list of [B, h_l], length L
        new_states: a list of (c=[B, h_l], h=[B, h_l], z=[B, 1]), length L
        """

        total_hidden_size = sum(c._h_above_size for c in self._cells)

        # split out the part of the input that stores values of ha
        raw_inp = inputs[:, :-total_hidden_size]                # [B, I]
        raw_h_aboves = inputs[:, -total_hidden_size:]           # [B, sum(ha_l)]

        ha_splits = [c._h_above_size for c in self._cells]
        h_aboves = array_ops.split(value=raw_h_aboves,
                                   num_or_size_splits=ha_splits, axis=1)

        z_below = tf.ones([tf.shape(inputs)[0], 1])             # [B, 1]
        raw_inp = array_ops.concat([raw_inp, z_below], axis=1)  # [B, I + 1]

        new_states = [0] * len(self._cells)
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                cur_state = state[i]    # ([B, h_l], [B, h_l], [B, 1])
                # i == 0: [B, I + 1] + [B, ha_l] -> [B, I + 1 + ha_l]
                # i != 0: [B, hb_l + 1] + [B, ha_l] -> [B, hb_l + 1 + ha_l]
                cur_inp = array_ops.concat(
                    [raw_inp, h_aboves[i]], axis=1, name='input_to_cell')

                raw_inp, new_state = cell(cur_inp, cur_state)
                new_states[i] = new_state

        hidden_states = [ns.h for ns in new_states]
        return hidden_states, new_states
