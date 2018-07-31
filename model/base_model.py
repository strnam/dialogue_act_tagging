import os
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
use_crf =True

NUM_CHECK_POINT = 5
evaluate_every = 100
batch_size = 16
num_epochs = 40
checkpoint_every = 100



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
      yield shuffled_data[start_index:end_index]


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


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self):
        """Defines self.config and self.logger
        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings
        """
        # self.config = config
        # self.logger = config.logger
        self.sess = None
        self.saver = None


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method='adam', lr=1e-3, clip=-1):
        """Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping
        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.grads_and_vars = optimizer.compute_gradients(self.loss)
                self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

    # def initialize_session(self):
    #     """Defines self.sess and initialize the variables"""
    #     print("Initializing tf session")
    #     self.sv = tf.train.Supervisor('/home/strnam/tmp')
    #     # self.sess = tf.Session()
    #     self.sess = self.sv.managed_session()
    #     self.sess.run(tf.global_variables_initializer())
    #     self.saver = tf.train.Saver(max_to_keep=NUM_CHECK_POINT)



    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            sess: tf.Session()
            dir_model: dir with weights
        """
        print("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    # def save_session(self):
    #     """Saves session = weights"""
    #     if not os.path.exists(self.config.dir_model):
    #         os.makedirs(self.config.dir_model)
    #     self.saver.save(self.sess, self.config.dir_model)

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
      if use_crf:
        self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        self.dev_summary_op = tf.summary.merge([loss_summary])
      else:
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])


      # Train Summaries
      train_summary_dir = os.path.join(out_dir, "summaries", "train")
      self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)
      dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
      self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

      # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
      checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
      self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # def train_step(self, x_batch, y_batch, x_sequence_lenght_batch):
    #   """
    #   A single training step
    #   """
    #   feed_dict = {
    #     self.input_x: x_batch,
    #     self.input_y: y_batch,
    #     self.session_lengths: x_sequence_lenght_batch,
    #   }
    #
    #   if use_crf:
    #     _, step, summaries, loss = self.sess.run(
    #       [self.train_op, self.global_step, self.train_summary_op, self.loss],
    #       feed_dict)
    #     time_str = datetime.now().isoformat()
    #     print("{}: step {}, loss {:g}".format(time_str, step, loss))
    #     self.train_summary_writer.add_summary(summaries, step)
    #   else:
    #     _, step, summaries, loss, accuracy = self.sess.run(
    #       [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy],
    #       feed_dict)
    #     time_str = datetime.now().isoformat()
    #     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    #     self.train_summary_writer.add_summary(summaries, step)
    #
    # def dev_step(self, x_batch, y_batch, x_sequence_lenght_batch, writer=None):
    #   """
    #   Evaluates model on a dev set
    #   """
    #   feed_dict = {
    #     self.input_x: x_batch,
    #     self.input_y: y_batch,
    #     self.session_lengths: x_sequence_lenght_batch,
    #   }
    #   if use_crf:
    #     viterbi_sequences = []
    #     step, summaries, loss, logits, trans_params = self.sess.run([self.global_step, self.dev_summary_op, self.loss,
    #                                                             self.scores, self.transition_params], feed_dict=feed_dict)
    #     for logit, sequence_length in zip(logits, x_sequence_lenght_batch):
    #       logit = logit[:sequence_length]  # keep only the valid steps
    #       viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
    #         logit, trans_params)
    #       viterbi_sequences += [viterbi_seq]
    #
    #     time_str = datetime.now().isoformat()
    #     y_batch_label = np.argmax(y_batch, axis=2)
    #     acc = compute_dialogue_act_acc(y_batch_label, viterbi_sequences, x_sequence_lenght_batch)
    #     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
    #   else:
    #     step, summaries, loss, accuracy = self.sess.run(
    #       [self.global_step, self.dev_summary_op, self.loss, self.accuracy],
    #       feed_dict)
    #     time_str = datetime.now().isoformat()
    #     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    #
    #   if writer:
    #     writer.add_summary(summaries, step)

    # def train(self, train, dev):
    #   x_train, y_train, x_seq_len_train = train
    #   if dev:
    #     x_dev, y_dev, x_seq_len_dev = dev
    #   batches = batch_iter(list(zip(x_train, y_train, x_seq_len_train)), batch_size=batch_size, num_epochs=num_epochs)

    #   self.add_summary()  # tensorboard
    #   # Training loop. For each batch...
    #   for batch in batches:
    #     x_batch, y_batch, x_sequence_lenght_batch = zip(*batch)
    #     self.train_step(x_batch, y_batch, x_sequence_lenght_batch)
    #     current_step = tf.train.global_step(self.sess, self.global_step)
    #     if dev:
    #       if current_step % evaluate_every == 0:
    #         print("\nEvaluation:")
    #         self.dev_step(x_dev, y_dev, x_seq_len_dev, writer=self.dev_summary_writer)
    #         print("")

    #     if current_step % checkpoint_every == 0:
    #       path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
    #       print("Saved model checkpoint to {}\n".format(path))

    # def evaluate(self, test):
    #     """Evaluate model on test set
    #     Args:
    #         test: instance of class Dataset
    #     """
    #     self.logger.info("Testing model over test set")
    #     metrics = self.run_evaluate(test)
    #     msg = " - ".join(["{} {:04.2f}".format(k, v)
    #             for k, v in metrics.items()])
    #     self.logger.info(msg)