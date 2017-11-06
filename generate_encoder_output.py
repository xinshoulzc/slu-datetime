# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def generate_embedding_RNN_output(encoder_inputs,
                            cell,
                            word_embedding_size,
                            dtype=tf.float32,
                            scope=None,
                            sequence_length=None,
                            bidirectional_rnn=False):
    with tf.variable_scope(scope or "generate_embedding_RNN_output"):
        # bidirectional_lstm was used
        # cell_fw forward cell
        # cell_bw backward cell
        if bidirectional_rnn:
            encoder_cell_fw = cell
            encoder_cell_bw = cell
            word_embedding_x = tf.get_variable("embedding_x", [word_embedding_size, word_embedding_size], dtype=tf.float32)
            word_embedding_b = tf.get_variable("embedding_b", [word_embedding_size], dtype=tf.float32)
            encoder_embedding_inputs = [tf.matmul(encoder_input, word_embedding_x) + word_embedding_b
                                        for encoder_input in encoder_inputs]
            # inputs size: [seq_len, batch_size, word_embedding_size]
            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.nn.static_bidirectional_rnn(
                encoder_cell_fw, encoder_cell_bw, encoder_embedding_inputs, sequence_length=sequence_length, dtype=dtype)
            # encoder_outputs size: [seq_length or timesteps, batch_size, state_length * 2(back and forward)]
            # encoder_state size: [batch_size, state_length * 2(back + forward)]
            encoder_state = tf.concat([tf.concat(encoder_state_fw, 1), tf.concat(encoder_state_bw, 1)], 1)
            top_states = [tf.reshape(e, [-1, 1, cell.output_size*2])
                          for e in encoder_outputs]
            # attention_state size: [batch, timesteps or seq_length, cell.output_size * 2]
            attention_states = tf.concat(top_states, 1)
        else:
            encoder_cell = cell
            word_embedding_x = tf.get_variable("embedding_x", [word_embedding_size, word_embedding_size],
                                               dtype=tf.float32)
            word_embedding_b = tf.get_variable("embedding_b", [word_embedding_size, word_embedding_size],
                                               dtype=tf.float32)
            encoder_embedding_inputs = [tf.matmul(encoder_input, word_embedding_x) + word_embedding_b
                                        for encoder_input in encoder_inputs]
            encoder_outputs, encoder_state = tf.nn.static_rnn(
                encoder_cell, encoder_embedding_inputs, sequence_length=sequence_length, dtype=dtype)
            encoder_state = tf.concat(encoder_state, 1)
            top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                          for e in encoder_outputs]
            attention_states = tf.concat(top_states, 1)
        # encoder_outputs size: [seq_length or timesteps, batch_size, state_length * 2(back and forward)]
        # encoder_state size: [batch_size, state_length * 2(back + forward)]
        # attention_state size: [batch, timesteps or seq_length, cell.output_size * 2]
        return encoder_outputs, encoder_state, attention_states