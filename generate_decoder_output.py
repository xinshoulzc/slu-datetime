# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops


def generate_decoder_output(encoder_outputs, encoder_state,
                            targets, sequence_length, buckets,
                            num_decoder_symbols, softmax_loss_function=None,
                            per_example_loss=None, name=None,
                            use_attention=False):
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    all_inputs = encoder_outputs + targets
    with tf.variable_scope(all_inputs, name, "model_with_buckets"):
        with tf.variable_scope("decoder_sequence_output", reuse=None):
            logits, attention_weight = attention_RNN(
                encoder_outputs, encoder_state, num_decoder_symbols,
                sequence_length, use_attention=use_attention)

        assert len(logits) == len(targets)
        # We need to make target and int64-tensor and set its shape.
        bucket_target = [tf.reshape(math_ops.to_int64(x), [-1]) for x in targets]
        # pay attention to return data between sequence_loss_by_example and sequence_loss
        if per_example_loss is None:
            crossent = sequence_loss_by_example(
                logits, bucket_target, softmax_loss_function=softmax_loss_function)
        else:
            crossent = sequence_loss(logits, bucket_target, softmax_loss_function=softmax_loss_function)

    return logits, crossent


def sequence_loss_by_example(logits, targets,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d." % (len(logits), len(targets)))
  with ops.op_scope(logits + targets, name,
                    "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target in zip(logits, targets):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = tf.reshape(target, [-1])
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent)
    log_perps = math_ops.add_n(log_perp_list)
  return log_perps




def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
    cost = tf.reduce_sum(sequence_loss_by_example(
        logits, targets, softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = tf.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)
    else:
      return cost


def attention_RNN(self, encoder_outputs,
                  encoder_state, num_decoder_symbols,
                  sequence_length, use_attention=True,
                  num_heads = 1, dtype=dtypes.float32,
                  scope=None):
    # num_heads: numbers of layers used in AM model
    if use_attention:
        print ('Use the attention RNN model')

        with tf.variable_scope(scope or "attention_RNN"):
            output_size = encoder_outputs[0].get_shape()[1].value
            top_states = [tf.reshape(e, [-1, 1, output_size]) for e in encoder_outputs]
            attention_states = tf.concat(top_states, 1)
            if not attention_states.get_shape()[1:2].is_fully_defined():
                raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                                 % attention_states.get_shape())

            batch_size = tf.shape(top_states[0])[0]
            attn_length = attention_states.get_shape()[1].value
            attn_size = attention_states.get_shape()[2].value

            # A perfect trick
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
            hidden_features = []
            v = []
            attention_vec_size = attn_size  # Size of query vectors for attention.
            for a in xrange(num_heads):
                k = tf.get_variable("AttnW_%d" % a,
                                                [1, 1, attn_size, attention_vec_size])
                hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                v.append(tf.get_variable("AttnV_%d" % a,
                                                     [attention_vec_size]))

            def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                attn_weights = []
                ds = []  # Results of attention reads will be stored here.
                for i in xrange(num_heads):
                    with tf.variable_scope("Attention_%d" % i):
                        # y is outputs of fully_connected layer
                        y = fully_connected(query, attention_vec_size)
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        # y [batch, 1, 1, attention_vec_size]
                        s = tf.reduce_sum(v[i] * tf.tanh(hidden_features[i] + y), [2, 3])
                        a = tf.nn.softmax(s)
                        attn_weights.append(a)
                        # Now calculate the attention-weighted vector d.
                        d = math_ops.reduce_sum(
                            tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                        ds.append(tf.reshape(d, [-1, attn_size]))
                return attn_weights, ds

            batch_attn_size = tf.stack([batch_size, attn_size])
            # batch_attn_size = [batch_size, attn_size]
            attns = [tf.zeros(batch_attn_size, dtype=dtype)
                     for _ in xrange(num_heads)]
            for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])

            # loop through the encoder_outputs
            attention_encoder_outputs = list()
            sequence_attention_weights = list()
            for i in xrange(len(num_decoder_symbols)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                if i == 0:
                    with tf.variable_scope("Initial_Decoder_Attention"):
                        initial_state = fully_connected(encoder_state, output_size)
                    attn_weights, ds = attention(initial_state)
                else:
                    attn_weights, ds = attention(encoder_outputs[i])
                output = tf.concat([ds[0], encoder_outputs[i]], 1)  # NOTE: here we temporarily assume num_head = 1
                with tf.variable_scope("AttnRnnOutputProjection"):
                    logit = fully_connected(output, num_decoder_symbols[i])
                attention_encoder_outputs.append(logit)  # NOTE: here we temporarily assume num_head = 1
                sequence_attention_weights.append(attn_weights[0])  # NOTE: here we temporarily assume num_head = 1
    else:
        print('Use the NON attention RNN model')
        with tf.variable_scope(scope or "non-attention_RNN"):
            attention_encoder_outputs = list()
            sequence_attention_weights = list()

            # copy over logits once out of sequence_length
            if encoder_outputs[0].get_shape().ndims != 1:
                (fixed_batch_size, output_size) = encoder_outputs[0].get_shape().with_rank(2)
            else:
                fixed_batch_size = encoder_outputs[0].get_shape().with_rank_at_least(1)[0]

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                batch_size = tf.shape(encoder_outputs[0])[0]
            if sequence_length is not None:
                sequence_length = math_ops.to_int32(sequence_length)
            if sequence_length is not None:  # Prepare variables
                zero_logit = tf.zeros(
                    tf.stack([batch_size, num_decoder_symbols]), encoder_outputs[0].dtype)
                zero_logit.set_shape(
                    tensor_shape.TensorShape([fixed_batch_size.value, num_decoder_symbols]))
                min_sequence_length = math_ops.reduce_min(sequence_length)
                max_sequence_length = math_ops.reduce_max(sequence_length)

            for time, input_ in enumerate(encoder_outputs):
                if time > 0: tf.get_variable_scope().reuse_variables()
                # pylint: disable=cell-var-from-loop
                # call_cell = lambda: cell(input_, state)
                generate_logit = lambda: fully_connected(encoder_outputs[time], num_decoder_symbols)
                # pylint: enable=cell-var-from-loop
                if sequence_length is not None:
                    logit = _step(
                        time, sequence_length, min_sequence_length, max_sequence_length, zero_logit, generate_logit)
                else:
                    logit = generate_logit
                attention_encoder_outputs.append(logit)

    return attention_encoder_outputs, sequence_attention_weights

def _step(time, sequence_length, min_sequence_length, max_sequence_length, zero_logit, generate_logit):
  # Step 1: determine whether we need to call_cell or not
  empty_update = lambda: zero_logit
  logit = control_flow_ops.cond(
      time < max_sequence_length, generate_logit, empty_update)

  # Step 2: determine whether we need to copy through state and/or outputs
  existing_logit = lambda: logit

  def copy_through():
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    copy_cond = (time >= sequence_length)
    return math_ops.select(copy_cond, zero_logit, logit)

  logit = control_flow_ops.cond(
      time < min_sequence_length, existing_logit, copy_through)
  logit.set_shape(zero_logit.get_shape())
  return logit
