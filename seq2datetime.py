import os
import random

import numpy as np
import tensorflow as tf

import data_utils
import model
import fasttext
import sys
import time

tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_integer("max_training_step", 10000, "Max traing steps")
tf.app.flags.DEFINE_string("train_dir", ".data/train", "Training directory")
tf.app.flags.DEFINE_integer("in_vocab_size", 20000, "max vocab size")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab size")
tf.app.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length.")
tf.app.flags.DEFINE_integer("max_target_length", 8, "Length of target sequence")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit)).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("fasttext_model", None, "Use Fasttext pre-trained word vector")
tf.app.flags.DEFINE_string("export_model", None, "Export trained model for serving")
tf.app.flags.DEFINE_integer("num_layer", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "Size of the word embedding")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.3, "dropout keep cell input and output prob.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True, "Use birectional RNN")
tf.app.flags.DEFINE_boolean("use_attention", True, "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_year_class", 40, "the number of year class (1980 to 2020)")
tf.app.flags.DEFINE_integer("max_month_class", 25, "the number of year class (-12 to 12)")
tf.app.flags.DEFINE_integer("max_day_class", 62, "the number of year class (-31 to 31)")
tf.app.flags.DEFINE_integer("max_week_class", 15, "the number of year class (-7 to 7)")


FLAGS = tf.app.flags.FLAGS

# fasttextmodel loading problem

_fasttext_model = None
if FLAGS.fasttext_model is not None and FLAGS.export_model is None:
    _fasttext_model = fasttext.load_model(FLAGS.fasttext_model)

# buckets meaning : length for each sentence
_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]

# num_decoder_symbol : the class number of decoder classification
_number_decoder_symbol = [(FLAGS.max_year_class, FLAGS.max_month_class,
                           FLAGS.max_day_class, FLAGS.max_week_class)]


def create_model(session, rev_vocab, source_vocab_size):
    with tf.variable_scope("model", reuse=None):
        model_train = model.Model(
            _fasttext_model, rev_vocab, source_vocab_size, _buckets, _number_decoder_symbol,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layer, FLAGS.max_gradient_norm, FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            use_attention=FLAGS.use_attention,
            forward_only=True)
    with tf.variable_scope("model", reuse=True):
        model_test = model.Model(
            _fasttext_model, rev_vocab, source_vocab_size, _buckets, _number_decoder_symbol,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layer, FLAGS.max_gradient_norm, FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            use_attention=FLAGS.use_attention,
            forward_only=True)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        print ("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model_train.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print ("Created model with fresh parameters")
        session.run(tf.initialize_all_variables())
    return model_train, model_test


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.
    """
    data_set = []
    with tf.gfile.GFile(source_path, mode='r') as source_file:
        with tf.gfile.GFile(target_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print ("Read data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                data_set.append([source_ids, target_ids])
                source, target = source_file.readline(), target_file.readline()
    # 2 outputs in each unit: source_ids, target_ids
    return data_set


def train():
    print ('Applying Parameters:')
    for k, v in FLAGS.__dict__['__flags'].iteritems():
        print ('%s: %s' % (k, str(v)))
    print ('Preparing data in %s' % FLAGS.data_dir)
    in_seq_train, out_seq_train, in_seq_dev, in_seq_test, vocab_path = \
        data_utils.get_vocab(FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)

    result_dir = FLAGS.train_dir + 'train/test_result'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    # make sure result is repeatable ???
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    sess = tf.Session()
    sess.as_default()

    #Create model.
    print("Max sequence length: %d." % _buckets[0][0])
    print("Creating %d layers of %d units." % (FLAGS.num_layer, FLAGS.size))

    model, model_test = create_model(sess, rev_vocab, len(vocab))
    print ("Creating model with source_vocab_size=%d" % len(rev_vocab))

    # Read data into buckets and compute their sizes.
    print ("Reading train data (train set limit: %d)." % FLAGS.max_train_data_size)
    train_set = read_data(in_seq_train, out_seq_train)

    # Train Loop
    step_time, loss = 0.0, 0.0
    current_step = 0

    best_valid_score, best_test_score = 0, 0
    train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/valid')

    while model.global_step.eval(sess) < FLAGS.max_traing_step:
        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, batch_sequence_length = model.get_batch(train_set)
        _, step_loss, summary, decoder_logits = model.get_input_feed(sess, encoder_inputs, _buckets, decoder_inputs)

        train_writer.add_summary(summary, current_step)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path)
            step_time, loss = 0.0, 0.0

    return sess, model_test, vocab




def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()




