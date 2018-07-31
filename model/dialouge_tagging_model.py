import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from .base_model import BaseModel
import numpy as np
import time
from datetime import datetime
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
# import layers as layers_lib
from . import layers as layers_lib

# num_hiden_units = 100
# max_utterance_in_session = 536
# max_word_in_utterance = 200
# num_classes = 43
# embedding_size = 100
# vocab_size = 10000
# utterance_filter_sizes = [3, 4, 5]
# utterance_num_filter = 8
# session_filter_sizes = [2, 3]
#
# session_num_filters = 8
# use_crf =True
#
# NUM_CHECK_POINT = 5
# evaluate_every = 100
# batch_size = 8
# num_epochs = 2
# checkpoint_every = 100

NUM_CHECK_POINT = 5




def batch_iter(data, batch_size, num_epochs, shuffle=True):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[shuffle_indices]
    else:
      shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      # yield shuffled_data[start_index:end_index]
      batch_data = shuffled_data[start_index:end_index]
      yield batch_data


def compute_dialogue_act_acc(labels, labels_pred, dialogue_lengths):
  correct_preds, total_correct, total_preds = 0., 0., 0.
  accs = []
  for lab, lab_pred, length in zip(labels, labels_pred,
                                   dialogue_lengths):
    lab = lab[:length]
    lab_pred = lab_pred[:length]
    accs += [ a==b for (a, b) in zip(lab, lab_pred)]

  acc = np.mean(accs)

  return acc


def tf_shift_utts_in_session(sessions, num_shift_row=2, max_utterance_in_session = 23):
  """
  Input
  [[1, 1, 1, 1],
  [1, 2, 2, 2],
  [1, 3, 3, 3],
  [1, 4, 4, 4],
  [1,5,5,5]]

  Output
  [[0 0 0 0]
  [0 0 0 0]
  [1 1 1 1]
  [1 2 2 2]
  [1 3 3 3]
  """
  diag_vec = [1]*(max_utterance_in_session - num_shift_row) + [0]*num_shift_row
  diag = tf.diag(diag_vec)
  diag = tf.cast(diag, dtype=tf.float32)
  zero_tail_session = tf.transpose(tf.matmul(sessions, diag, transpose_a=True))
  zero_indices = [max_utterance_in_session - i - 1 for i in range(num_shift_row)]
  new_indices = tf.constant(zero_indices + list(range(0, max_utterance_in_session - num_shift_row)), dtype=tf.int32)
  zero_head_session = tf.gather(zero_tail_session, new_indices)
  return zero_head_session


def metric_absolute_match(y_true, y_preds):
  y_true_argmax = np.argmax(y_true, axis=1)
  y_pred_argmax = np.argmax(y_preds, axis=1)
  return accuracy_score(y_true_argmax, y_pred_argmax)


def f1(y_true, y_preds):
  f1_l = []
  for yt, yp in zip(y_true, y_preds):
    f1_l.append(f1_score(yt, yp, average=None))
  return np.mean(f1_l)


def detail_f1_score(y_true, y_preds):
  num_classes = y_true.shape[1]
  f1_l = []
  for i in range(num_classes):
    yt = y_true.T[i]
    yp = y_preds.T[i]
    f1_l.append(f1_score(yt, yp))
  return f1_l


def detail_precision(y_true, y_preds):
  num_classes = y_true.shape[1]
  precision_l = []
  for i in range(num_classes):
    yt = y_true.T[i]
    yp = y_preds.T[i]
    precision_l.append(precision_score(yt, yp))
  return precision_l

def detail_recall(y_true, y_preds):
  num_classes = y_true.shape[1]
  recall_l = []
  for i in range(num_classes):
    yt = y_true.T[i]
    yp = y_preds.T[i]
    recall_l.append(recall_score(yt, yp))
  return recall_l

def detail_accuracy_score(y_true, y_preds):
  num_classes = y_true.shape[1]
  acc_l = []
  for i in range(num_classes):
    yt = y_true.T[i]
    yp = y_preds.T[i]
    acc_l.append(accuracy_score(yt, yp))
  return acc_l


def metric_just_match(y_true, y_preds):
  match = 0
  for yt, yp in zip(y_true, y_preds):
    if any((yt + yp) == 2):
      match += 1
  # print(match)
  return match/len(y_true)



def evaluate(preds, threshold=0.5):
  preds[preds >= threshold] = 1
  preds[preds < threshold] = 0
  score = metric_absolute_match(np.array(y_test_vec_flat), preds)
  f1_s = f1(np.array(y_test_vec_flat), preds)
  print("Accuracy: ", score)
  print("F1: ", f1_s)


def build_word_matrix(word_index, embedding_pretrain, vocab_size, embedding_size):
  embedding_matrix = np.zeros((vocab_size, embedding_size))
  for word, i in word_index.items():
    if i >= vocab_size: 
      continue
    try:
      embedding_vector = embedding_pretrain[word]
    except:
      print("%s is not in word embedding" % word)
      embedding_vector = None
    # embedding_vector = embedding_pretrain.get(word)
    if embedding_vector is not None: 
      embedding_matrix[i] = embedding_vector
  return embedding_matrix

def restore_pretrained_model(sess, saver_for_restore, model_dir):
  """Restores pretrained model if there is no ckpt model."""
  pretrain_ckpt = tf.train.get_checkpoint_state(model_dir)
  if not (pretrain_ckpt and pretrain_ckpt.model_checkpoint_path):
    raise ValueError(
        'Asked to restore model from %s but no checkpoint found.' % model_dir)
  saver_for_restore.restore(sess, pretrain_ckpt.model_checkpoint_path)


class DialougeTagging(BaseModel):
  def __init__(self):
    self.sess = None
    self.saver = None

    self.num_hiden_units = 100
    self.max_utterance_in_session = 536
    self.max_word_in_utterance = 200
    self.num_classes = 43
    self.embedding_size = 100
    self.vocab_size = 10000
    self.utterance_filter_sizes = [3, 4, 5]
    self.utterance_num_filter = 8
    self.session_filter_sizes = [2, 3]
    self.session_num_filters = 8
    self.use_crf = False

    self.NUM_CHECK_POINT = 5
    self.evaluate_every = 100
    self.batch_size = 12
    self.num_epochs = 1
    self.checkpoint_every = 100
    self.use_pretrained_word_embedding = False
    self.pretrained_word_matrix = None

  def reinitialize_weights(self, scope_name):
    """Reinitializes the weights of a given layer"""
    variables = tf.contrib.framework.get_variables(scope_name)
    init = tf.variables_initializer(variables)
    self.sess.run(init)


  def restore_session(self, dir_model):
    """Reload weights into session
    Args:
        sess: tf.Session()
        dir_model: dir with weights
    """
    print("Reloading the latest trained model...")
    self.build_model()
    self.add_summary()  # t#ensorboard
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    self.saver = tf.train.Saver(max_to_keep=NUM_CHECK_POINT)
    self.saver.restore(self.sess, dir_model)


  def close_session(self):
    """Closes the session"""
    self.sess.close()

  def add_summary(self):
    # Difine dir
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in self.grads_and_vars:
      if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", self.loss)

    # Dev summaries
    if self.use_crf:
      self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
      self.dev_summary_op = tf.summary.merge([loss_summary])
    else:
      self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
      self.dev_summary_op = tf.summary.merge([loss_summary])


    # Train Summaries
    self.outdir = out_dir

    self.train_summary_dir = os.path.join(out_dir, "summaries", "train")
    self.dev_summary_dir = os.path.join(out_dir, "summaries", "dev")

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    # os.mkdir(os.path.join(out_dir, "checkpoints"))
    checkpoint_dir = os.path.abspath(out_dir)
    self.checkpoint_prefix = os.path.join(self.train_summary_dir, "model.ckpt")

    config = projector.ProjectorConfig()


    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

  def train_step(self, x_batch, y_batch, role_batch, x_sequence_lenght_batch, pretrained_word_embedd=None):
    """
    A single training step
    """
    feed_dict = {
      self.input_x: x_batch,
      self.input_y: y_batch,
      self.input_role: role_batch,
      self.session_lengths: x_sequence_lenght_batch,
      self.keep_prob: 0.9
      # self.embedding_placeholder: self.pretrained_word_matrix
    }

    if self.use_crf:
      _, step, summaries, loss = self.sess.run(
        [self.train_op, self.global_step, self.train_summary_op, self.loss],
        feed_dict)
      time_str = datetime.now().isoformat()
      print("{}: step {}, loss {:g}".format(time_str, step, loss))
      self.train_summary_writer.add_summary(summaries, step)
    else:
      _, step, summaries, loss, preds, y_true = self.sess.run(
        [self.train_op, self.global_step, self.train_summary_op, self.loss, self.pred, self.y_true],
        feed_dict)
      time_str = datetime.now().isoformat()
      score = metric_absolute_match(np.array(y_true), preds)
      f1_s = f1(np.array(y_true), preds)
      right = metric_just_match(y_true, preds)
      print("{}: step {}, loss {:g}, acc {:g}, f1 {}, just_right {:g}".format(time_str, step, loss, score, f1_s, right))
      self.train_summary_writer.add_summary(summaries, step)

  def predict(self, x_batch, role_batch, x_sequence_lenght_batch):
    feed_dict = {
      self.input_x: x_batch,
      self.input_role: role_batch,
      self.session_lengths: x_sequence_lenght_batch,
      # self.embedding_placeholder: self.pretrained_word_matrix

    }

    score = self.sess.run(
      [self.scores],
      feed_dict)
    time_str = datetime.now().isoformat()
    return score

  def dev_step(self, x_batch, y_batch, role_batch, x_sequence_lenght_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      self.input_x: x_batch,
      self.input_y: y_batch,
      self.input_role: role_batch,
      self.session_lengths: x_sequence_lenght_batch,
      # self.embedding_placeholder: self.pretrained_word_matrix

    }
    if self.use_crf:
      viterbi_sequences = []
      step, summaries, loss, logits, trans_params = self.sess.run([self.global_step, self.dev_summary_op, self.loss,
                                                                   self.scores, self.transition_params],
                                                                  feed_dict=feed_dict)
      for logit, sequence_length in zip(logits, x_sequence_lenght_batch):
        logit = logit[:sequence_length]  # keep only the valid steps
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
          logit, trans_params)
        viterbi_sequences += [viterbi_seq]

      time_str = datetime.now().isoformat()
      y_batch_label = np.argmax(y_batch, axis=2)
      acc = compute_dialogue_act_acc(y_batch_label, viterbi_sequences, x_sequence_lenght_batch)
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
    else:
      step, summaries, loss,  preds, y_true = self.sess.run(
        [self.global_step, self.dev_summary_op, self.loss,self.pred, self.y_true],
        feed_dict)
      time_str = datetime.now().isoformat()
      score = metric_absolute_match(np.array(y_true), preds)
      f1_s = f1(np.array(y_true), preds)
      # right = metric_just_match(y_true, preds)
      f1_detail = detail_f1_score(y_true, preds)
      precision_detail = detail_precision(y_true, preds)
      recall_detail = detail_recall(y_true, preds)
      acc_detail = detail_accuracy_score(y_true, preds)
      print("F1 detail: ", f1_detail)
      print("Accuracy detail: ", acc_detail)
      print("Precision detail: ", precision_detail)
      print("Recall detail: ", recall_detail)
    if writer:
      writer.add_summary(summaries, step)
    
    return score

  def train(self, train, dev=None, test=None):
    x_train, y_train, role_train, x_seq_len_train = train
    if dev:
      x_dev, y_dev, role_dev, x_seq_len_dev = dev
    batches = batch_iter(list(zip(x_train, y_train, role_train, x_seq_len_train)), batch_size=self.batch_size, num_epochs=self.num_epochs)

    ### New session
    local_init_op = tf.train.Supervisor.USE_DEFAULT
    ready_for_local_init_op = tf.train.Supervisor.USE_DEFAULT
    is_chief = True
    
    self.build_model()
    self.add_summary()  # t#ensorboard
    self.saver = tf.train.Saver(max_to_keep=NUM_CHECK_POINT)
    # import ipdb;ipdb.set_trace()
    saver_for_restore = tf.train.Saver(self.pretrained_variables)

    self.sv = tf.train.Supervisor(
        # logdir='/home/strnam/tmp',
        is_chief=is_chief,
        # save_summaries_secs=15,
        # save_model_secs=15,
        local_init_op=local_init_op,
        ready_for_local_init_op=ready_for_local_init_op,
        global_step=self.global_step,
        summary_op=None)
    
    with self.sv.managed_session( master='',
      config=tf.ConfigProto(log_device_placement=False),
      start_standard_services=False) as sess:
        # Init
      pretrain_model_path = '/media/strnam/New Volume/research/models/research/pretrain/models/gotit_pretrain_1'
      restore_pretrained_model(sess, saver_for_restore, pretrain_model_path)
      self.sess = sess
      self.sv.start_standard_services(sess)
      self.sv.start_queue_runners(sess)

      ## Add summary

      self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, self.sess.graph)
      self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir, self.sess.graph)

      # Training loop. For each batch...
      not_improve = 0
      best_accuracy = 0
      for batch in batches:
        x_batch, y_batch, role_batch, x_sequence_lenght_batch = zip(*batch)
        self.train_step(x_batch, y_batch, role_batch, x_sequence_lenght_batch)
        current_step = tf.train.global_step(self.sess, self.global_step)
        if dev:
          if current_step % self.evaluate_every == 0:
            # continue
            print("\nEvaluation:")
            cur_accuracy = self.dev_step(x_dev, y_dev, role_dev, x_seq_len_dev, writer=self.dev_summary_writer)
            print("Dev accuracy: %s" % cur_accuracy)
            if cur_accuracy > best_accuracy:
              not_improve = 0
              print("New best accuracy: %s" % cur_accuracy)
              best_accuracy = cur_accuracy
              # path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
              path = self.saver.save(self.sess, self.checkpoint_prefix)
              print("Saved best model checkpoint to {}\n".format(path))
            else:
              not_improve += 1
              print("Accuracy does not improve, continue")
          if not_improve > 5:
            print('Early stopping')
            break

        if current_step % self.checkpoint_every == 0:
          # continue
          path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
          print("Saved model checkpoint to {}\n".format(path))

      if test:
        x_test, y_test, role_test, x_seq_len_test = test
        print('==--'*5)
        print('TEST')
        cur_accuracy2 = self.dev_step(x_test, y_test, role_test, x_seq_len_test)
        print('Accuracy test', cur_accuracy2)


  def build_model(self):
    self.add_input()
    self.add_embedding()
    self.add_utterance_lstm_model()
    # self.add_utterance_model()
    # self.add_disclosure_model()
    self.add_classification_layer()
    self.add_loss_op()
    self.add_accuracy_op()
    self.add_train_op()
    # self.initialize_session()

  def add_train_op(self, lr_method='adam', lr=1e-3, clip=-1):
    """Defines self.train_op that performs an update on a batch
    Args:
        lr_method: (string) sgd method, for example "adam"
        lr: (tf.placeholder) tf.float32, learning rate
        loss: (tensor) tf.float32 loss to minimize
        clip: (python float) clipping of gradient. If < 0, no clipping
    """
    _lr_m = lr_method.lower()  # lower to make sure

    with tf.variable_scope("train_step"):
      if _lr_m == 'adam':  # sgd method
        optimizer = tf.train.AdamOptimizer(lr)
      elif _lr_m == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr)
      elif _lr_m == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
      elif _lr_m == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(lr)
      else:
        raise NotImplementedError("Unknown method {}".format(_lr_m))

      if clip > 0:  # gradient clipping if clip is positive
        grads, vs = zip(*optimizer.compute_gradients(self.loss))
        grads, gnorm = tf.clip_by_global_norm(grads, clip)
        self.train_op = optimizer.apply_gradients(zip(grads, vs))
      else:
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

   
  
  @property
  def pretrained_variables(self):
    return (self.layer_embedding.trainable_weights +
            self.layer_lstm.trainable_weights)


  def add_input(self):
    # Init Variable
    self.input_x = tf.placeholder(tf.int32, shape=[None, self.max_utterance_in_session, self.max_word_in_utterance], name="input_x_raw")
    self.input_y = tf.placeholder(tf.float32, [None, self.max_utterance_in_session, self.num_classes], name="input_y")
    self.input_role = tf.placeholder(tf.float32, [None, self.max_utterance_in_session], name="input_role")
    self.session_lengths = tf.placeholder(tf.int32, [None], name="session_lenght")
    self.word_ids = tf.reshape(self.input_x, [-1, self.max_word_in_utterance])
    self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
    self.keep_prob = tf.placeholder_with_default(1.0, shape=())


  def add_embedding(self):
        # Word Embedding layer
    # with tf.device('/cpu:0'), tf.name_scope("embedding"):
    self.layer_embedding = layers_lib.Embedding(
      self.vocab_size, self.embedding_size)
    self.word_embedding_raw = self.layer_embedding(self.word_ids)


  def add_utterance_lstm_model(self):
    with tf.name_scope('utterance-lstm-maxpool'):
      # rnn_cell_size = 1024
      rnn_cell_size = 512
      rnn_num_layers = 1
      keep_prob_lstm_out = 1.0
      self.layer_lstm = layers_lib.LSTM(
        rnn_cell_size, rnn_num_layers, keep_prob_lstm_out)

      lstm_output, _ = self.layer_lstm(self.word_embedding_raw, None, None)

      print("lstm output shape: ", lstm_output.get_shape())

      lstm_output_expand = tf.expand_dims(lstm_output, -1)

      #lstm_output_expand: [ , max_word_in_utterance, embedding_size, 1]
      lstm_output_pooled = tf.nn.max_pool(
          lstm_output_expand,
          ksize=[1, self.max_word_in_utterance, 1, 1],
          strides=[1, 1, 1, 1],
          padding="VALID"
        )
      print("lstm pooled shape: ", lstm_output_pooled.get_shape())
      #lstm_output_pooled: [ , 1, embedding_size, 1]
      self.num_utterance_filters_total = lstm_output.get_shape().as_list()[-1]
      self.utterance_vector_group_by_sess = tf.reshape(lstm_output_pooled, [-1, self.max_utterance_in_session, self.num_utterance_filters_total])
  
  def add_utterance_model(self):
    # Utterance representation
    self.word_embedding = tf.expand_dims(self.word_embedding_raw, -1)
    print("embedding re-dim: ", self.word_embedding.get_shape())
    applied_conv_on_sentence_outputs = []
    for i, filter_size in enumerate(self.utterance_filter_sizes):
      with tf.name_scope('utterance-conv-maxpool-%s' % filter_size):
        filter_shape = [filter_size, self.embedding_size, 1, self.utterance_num_filter]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[self.utterance_num_filter]), name="b")
        conv = tf.nn.conv2d(
          self.word_embedding,
          W,
          strides=[1, 1, 1, 1],
          padding="VALID",
          name="conv")
        # Apply nonlinearity
        print(conv.get_shape())
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #Maxpooling over the outputs.
        pooled = tf.nn.max_pool(
          h,
          ksize=[1, self.max_word_in_utterance - filter_size + 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding="VALID"
        )
        # pooled has shape: [batch*max_utterance_in_session, 1, 1, utterance_num_filter]
        applied_conv_on_sentence_outputs.append(pooled)

    self.num_utterance_filters_total = self.utterance_num_filter*len(self.utterance_filter_sizes)
    utterance_vector = tf.concat(applied_conv_on_sentence_outputs, 3)

    self.utterance_vector_group_by_sess = tf.reshape(utterance_vector, [-1, self.max_utterance_in_session, self.num_utterance_filters_total])


  def add_classification_layer(self):
    utterance_vector_length = self.num_utterance_filters_total
    with tf.name_scope("Classification"):
      # Layer 1
      role_1_layer1_W_minus_2 = tf.get_variable("role_1_layer1_W_2", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      role_1_layer1_W_minus_1 = tf.get_variable("role_1_layer1_W_1", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      role_1_layer1_W = tf.get_variable("role_1_layer1_W", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      role_1_layer1_b = tf.get_variable("role_1_layer1_b", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())


      role_2_layer1_W_minus_2 = tf.get_variable("role_2_layer1_W_2", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      role_2_layer1_W_minus_1 = tf.get_variable("role_2_layer1_W_1", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      role_2_layer1_W = tf.get_variable("role_2_layer1_W", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      role_2_layer1_b = tf.get_variable("role_2_layer1_b", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())

      # Prepare the filter for role:
      role_data = tf.reshape(self.input_role, shape=[-1])
      role_1_on = (role_data + 1)/2
      role_2_on = (role_data - 1)/2


      # Layer 2
      layer2_W_minus_2 = tf.get_variable("layer2_W_2", shape=[self.num_classes, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer2_W_minus_1 = tf.get_variable("layer2_W_1", shape=[self.num_classes, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer2_W = tf.get_variable("layer2_W", shape=[self.num_classes, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())

      layer2_b = tf.get_variable("layer2_b", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())

      # TODO: need to padding zero and remove the last row in minus 2 and minus 1 matrix.
      num_sess = self.input_x.shape[0]
      sess_mat_minus2 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=2, max_utterance_in_session=self.max_utterance_in_session),
                                  self.utterance_vector_group_by_sess)
      sess_mat_minus1 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=1, max_utterance_in_session=self.max_utterance_in_session),
                                  self.utterance_vector_group_by_sess)


      sess_mat_minus2_flat = tf.reshape(sess_mat_minus2, [-1, utterance_vector_length])
      sess_mat_minus1_flat = tf.reshape(sess_mat_minus1, [-1, utterance_vector_length])
      sess_mat_flat = tf.reshape(self.utterance_vector_group_by_sess, [-1, utterance_vector_length])

      # Role 1
      role_1_layer1_mat_minus_2 = tf.multiply(tf.matmul(sess_mat_minus2_flat, role_1_layer1_W_minus_2), role_1_on[:,tf.newaxis])
      role_1_layer1_mat_minus_1 = tf.multiply(tf.matmul(sess_mat_minus1_flat, role_1_layer1_W_minus_1), role_1_on[:,tf.newaxis])
      role_1_layer1_mat_mul =  tf.multiply(tf.matmul(sess_mat_flat, role_1_layer1_W), role_1_on[:,tf.newaxis])
      role_1_layer1_b = tf.matmul(role_1_on[:,tf.newaxis], role_1_layer1_b[tf.newaxis,:])


      # Role 2
      role_2_layer1_mat_minus_2 = tf.multiply(tf.matmul(sess_mat_minus2_flat, role_2_layer1_W_minus_2), role_2_on[:,tf.newaxis])
      role_2_layer1_mat_minus_1 = tf.multiply(tf.matmul(sess_mat_minus1_flat, role_2_layer1_W_minus_1), role_2_on[:,tf.newaxis])
      role_2_layer1_mat_mul =  tf.multiply(tf.matmul(sess_mat_flat, role_2_layer1_W), role_2_on[:,tf.newaxis])
      role_2_layer1_b = tf.matmul(role_2_on[:, tf.newaxis], role_2_layer1_b[tf.newaxis, :])

      layer1_mat_minus_2 = role_1_layer1_mat_minus_2 + role_2_layer1_mat_minus_2
      layer1_mat_minus_1 = role_1_layer1_mat_minus_1 + role_2_layer1_mat_minus_1
      layer1_mat_mul = role_1_layer1_mat_mul + role_2_layer1_mat_mul
      layer1_b = role_1_layer1_b + role_2_layer1_b

      y_layer1 = tf.tanh(layer1_mat_minus_2 + layer1_mat_minus_1 + layer1_mat_mul + layer1_b)

      y_layer1_reshape = tf.reshape(layer1_mat_minus_2, [-1, self.max_utterance_in_session, self.num_classes])


      layer2_sess_mat_minus2 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=2,
                                                                                    max_utterance_in_session=self.max_utterance_in_session),
                                         y_layer1_reshape)
      layer2_sess_mat_minus1 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=1,
                                                                                    max_utterance_in_session=self.max_utterance_in_session),
                                         y_layer1_reshape)


      layer2_sess_mat_minus2_flat = tf.reshape(layer2_sess_mat_minus2, [-1, self.num_classes])
      layer2_sess_mat_minus1_flat = tf.reshape(layer2_sess_mat_minus1, [-1, self.num_classes])
      layer2_sess_mat_flat = y_layer1

      layer2_mat_minus_2 = tf.matmul(layer2_sess_mat_minus2_flat, layer2_W_minus_2)
      layer2_mat_minus_1 = tf.matmul(layer2_sess_mat_minus1_flat, layer2_W_minus_1)
      layer2_mat_mul = tf.matmul(layer2_sess_mat_flat, layer2_W)
      pred = (layer2_mat_minus_2 + layer2_mat_minus_1 + layer2_mat_mul + layer2_b)
      self.scores = tf.reshape(pred, [-1, self.max_utterance_in_session, self.num_classes])
      print('self.scores', self.scores.get_shape())

      self.mask = tf.sequence_mask(self.session_lengths, maxlen=self.max_utterance_in_session)

  def add_loss_op(self):
    with tf.name_scope("loss"):
      if self.use_crf:
        labels = tf.argmax(self.input_y, axis=2)
        labels = tf.cast(labels, dtype=tf.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
          self.scores, labels, self.session_lengths)
        self.transition_params = transition_params
        self.loss = tf.reduce_mean(-log_likelihood)
      else:
        self.scores = tf.boolean_mask(self.scores, self.mask)
        self.y_true = tf.boolean_mask(self.input_y, self.mask)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_true)

        self.loss = tf.reduce_mean(losses)


  def add_accuracy_op(self):
    if not self.use_crf:
      with tf.name_scope("Accuracy"):
        # self.predictions = tf.boolean_mask(self.scores, self.mask)
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
        self.pred = tf.one_hot(self.predictions, self.num_classes)
        # self.predictions = tf.argmax(self.predictions, 1, name="predictions")
        y_by_utterance = tf.boolean_mask(self.input_y, self.mask)
        self.y_arg_max = tf.argmax(y_by_utterance, 1)
        print('yarg_max', self.y_arg_max.get_shape())
        correct_predictions = tf.equal(self.predictions, self.y_arg_max)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





