import tensorflow as tf
import numpy as np
import generate_encoder_output
import generate_decoder_output

import random
import data_utils


class Model(object):
    def __init__(self, fasttext_model, rev_vocab, source_vocab_size, buckets, number_decoder_symbol,
                 word_embedding_size, size, num_layers, max_gradient_norm, batch_size,
                 dropout_keep_prob=1.0, use_lstm=False, bidirectional_rnn=True,
                 use_attention=False, forward_only=False):
        self.fasttext_model = fasttext_model
        self.rev_vocab = rev_vocab
        self.source_vocab_size = source_vocab_size
        self.buckets = buckets
        self.number_decoder_symbol = number_decoder_symbol
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver()

        softmax_loss_function = None

        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        if not forward_only and dropout_keep_prob < 1.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 input_keep_prob=dropout_keep_prob,
                                                 output_keep_prob=dropout_keep_prob)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")

        for i in xrange(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, word_embedding_size],
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][0]):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[None], name="decoder{0}".format(i)))

        encoder_outputs, encoder_state, attention_states = \
            generate_encoder_output.generate_embedding_RNN_output(self.encoder_inputs,
                                                                               cell,
                                                                               word_embedding_size,
                                                                               dtype=tf.float32,
                                                                               scope=None,
                                                                               sequence_length=self.sequence_length,
                                                                               bidirectional_rnn=bidirectional_rnn)
        self.datetime_output, self.datetime_loss = \
            generate_decoder_output.generate_decoder_output(
            encoder_state, attention_states,
            self.sequence_length, self.buckets, self.number_decoder_symbol,
            softmax_loss_function=softmax_loss_function,
            use_attention=use_attention)

        self.saver = tf.train.Saver()

    def get_batch(self, data):
        encoder_size, decoder_size = self.buckets[0]
        encoder_inputs, decoder_inputs = [], []
        batch_sequence_length_list = list()
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data)
            batch_sequence_length_list.append(len(encoder_input))

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(encoder_input + encoder_pad))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input)
            decoder_inputs.append(list(decoder_input + [data_utils.PAD_ID] * decoder_pad_size))

        # create batch-major vectors from the data selected above
        batch_encoder_inputs, batch_decoder_inputs = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            encoder_batch, decoder_batch = [], []
            for batch_idx in xrange(self.batch_size):
                word = self.rev_vocab[encoder_inputs[batch_idx][length_idx]]
                word_vec = self.fasttext_model[word]
                encoder_batch.append(word_vec)
                decoder_batch.append(decoder_inputs[batch_idx][length_idx])
            batch_encoder_inputs.append(np.array(encoder_batch, dtype=np.float32))
            batch_decoder_inputs.append(np.array(decoder_batch, dtype=np.float32))

        batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
        return batch_encoder_inputs, batch_decoder_inputs, batch_sequence_length

    def get_input_feed(self, session, encoder_outputs, batch_sequence_length,
                       buckets, decoder_inputs,
                 forword_only):
        encoder_size, decoder_size = buckets[-1]
        if len(encoder_outputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_outputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
        all_inputs = encoder_outputs + decoder_inputs
        # Input feed: encoder inputs, decoder_inputs
        input_feed = dict({})
        input_feed[self.sequence_length.name] = batch_sequence_length
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_outputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        if not forword_only:
            output_feed = [
                self.update,  # Update Op that does SGD.
                self.gradient_norm,  # Gradient norm.
                self.loss,
                self.summary  # Loss for this batch.
            ]
            for i in xrange(decoder_size):
                output_feed.append(self.target_output[i])
        else:
            output_feed = [self.loss]
            for i in xrange(decoder_size):
                output_feed.append(self.target_output[i])
        outputs = session.run(output_feed, input_feed)
        if not forword_only: return outputs[1], outputs[2], outputs[3], outputs[4:4+decoder_size]
        else: return None, outputs[0], outputs[1:1+decoder_size]

