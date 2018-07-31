from data_provider import dialogue_dataset
from build_data import Config
import os
import pickle
import numpy as np
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, TimeDistributed, Dense, Activation, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from model.dialouge_tagging_model import DialougeTagging


#DATASET_NAME = 'Switchboard'
DATASET_NAME = 'Gotit'
PRETRAIN_DIR = './pre_train'
EMBEDDING_FILE = '/media/strnam/New Volume/competition/Toxic_comment/fasttext/crawl-300d-2M.vec'
# EMBEDDING_FILE = '/media/strnam/New Volume/research/thesis/w2v.txt'
MAX_VOCAB_SIZE = 30000
EMBEDDING_SIZE = 300



class CustomTokenizer(object):
  def __init__(self, vocab_file_path):
    self.word_index = self.make_vocab_ids(vocab_file_path)
  
  def texts_to_sequences(self, text_seq):
    def tokens_to_ids(tokens):
      return [self.word_index[tok] for tok in tokens if tok in self.word_index]
    return [tokens_to_ids(text.split()) for text in text_seq]

  def make_vocab_ids(self, vocab_filename):
    with open(vocab_filename) as vocab_f:
      return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])
 

def simple_classification_non_mutual():
  # LOAD DATA
  print('Load data')
  config = Config(DATASET_NAME)
  dataset = dialogue_dataset.get_dataset(DATASET_NAME)
  num_tags = dataset.get_num_tags()

  x_train, y_train = dataset.get_dialogue_data(config.train_ids)  # list[list], list[list]
  role_train = dataset.get_role_data(config.train_ids)
  x_test, y_test = dataset.get_dialogue_data(config.test_ids)
  role_test = dataset.get_role_data(config.test_ids)

  x_dev, y_dev = dataset.get_dialogue_data(config.valid_ids)
  role_dev = dataset.get_role_data(config.valid_ids)



  # Store dialouge lenght
  x_train_dialogue_len = dataset.get_dialogues_length(x_train)
  x_test_dialogue_len = dataset.get_dialogues_length(x_test)
  x_dev_dialogue_len = dataset.get_dialogues_length(x_dev)
  # TRAIN UTT2VEC MODEL
  print('Train Utterance to Vector')

  # utt2vec_model = Utt2vec()

  x_train_flat = dataset.flat_dialogue(x_train)
  x_test_flat = dataset.flat_dialogue(x_test)
  x_dev_flat = dataset.flat_dialogue(x_dev)


  len_sent = [len(sent.split()) for sent in x_train_flat]
  # len_sent = [len(sent_toks) for sent_toks in x_train_flat]
  max_length = np.max(len_sent)
  max_utt_in_sess = max(max(x_train_dialogue_len), max(x_test_dialogue_len))

  vocab_pretrain = '/media/strnam/New Volume/research/models/research/pretrain/data/gotit/vocab.txt'
  tokenizer = CustomTokenizer(vocab_pretrain)

  X_train = tokenizer.texts_to_sequences(x_train_flat)
  X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
  X_train = dataset.group_utterance_to_dialogue(X_train, x_train_dialogue_len)
  X_train = pad_sequences(X_train, maxlen=max_utt_in_sess, padding='post')

  X_test = tokenizer.texts_to_sequences(x_test_flat)
  X_test = pad_sequences(X_test, maxlen=max_length, padding='post')
  X_test = dataset.group_utterance_to_dialogue(X_test, x_test_dialogue_len)
  X_test = pad_sequences(X_test, maxlen=max_utt_in_sess, padding='post')

  X_dev = tokenizer.texts_to_sequences(x_dev_flat)
  X_dev = pad_sequences(X_dev, maxlen=max_length, padding='post')
  X_dev = dataset.group_utterance_to_dialogue(X_dev, x_dev_dialogue_len)
  X_dev = pad_sequences(X_dev, maxlen=max_utt_in_sess, padding='post')


  # y_train: num_dialogue, num_utt, num_tags
  def to_categorical_non_mutual(y_train, num_tags):
    # y_train_tag_one_hot = [to_categorical(diag, num_classes=num_tags) for diag in y_train]
    y_train_categorical_vec = []
    for y_diag in y_train:
      y_train_categorical_vec_diag = []
      for y_utt in y_diag:
        y_utt_one_hot_vecs = to_categorical(y_utt, num_tags)
        tags_utt_vec = np.sum(y_utt_one_hot_vecs, axis=0)
        y_train_categorical_vec_diag.append(tags_utt_vec)
      y_train_categorical_vec.append(y_train_categorical_vec_diag)
    return y_train_categorical_vec

  y_train_vec = to_categorical_non_mutual(y_train, num_tags)
  y_train_vec_flat = np.concatenate(y_train_vec, axis=0).tolist()
  y_train_pad = pad_sequences(y_train_vec, maxlen=max_utt_in_sess, padding='post')
  role_train_pad = pad_sequences(role_train, maxlen=max_utt_in_sess, padding='post', value=-1)

  y_dev_vec = to_categorical_non_mutual(y_dev, num_tags)
  y_dev_vec_flat = np.concatenate(y_dev_vec, axis=0).tolist()
  y_dev_pad = pad_sequences(y_dev_vec, maxlen=max_utt_in_sess, padding='post')
  role_dev_pad = pad_sequences(role_dev, maxlen=max_utt_in_sess, padding='post', value=-1)

  y_test_vec = to_categorical_non_mutual(y_test, num_tags)
  y_test_vec_flat = np.concatenate(y_test_vec, axis=0).tolist()
  y_test_pad = pad_sequences(y_test_vec, max_utt_in_sess, padding='post')
  role_test_pad = pad_sequences(role_test, maxlen=max_utt_in_sess, padding='post', value=-1)

  num_vocabs = 10000

  model = DialougeTagging()
  model.max_word_in_utterance = max_length
  model.max_utterance_in_session = max(x_train_dialogue_len)
  model.num_classes = dataset.get_num_tags()
  model.evaluate_every = 5
  model.num_epochs = 50
  model.vocab_size = len(tokenizer.word_index) #86934 # vocab_size
  model.embedding_size = 100 #EMBEDDING_SIZE
  # model.pretrained_word_matrix = embedding_matrix
  model.pretrained_word_matrix = None


  model.utterance_filter_sizes = [2,3]
  model.utterance_num_filter = 2
  model.session_filter_sizes = [2, 3]
  model.session_num_filters = 2
  model.checkpoint_every = 10

  train = (X_train, y_train_pad, role_train_pad, x_train_dialogue_len)
  test = (X_test, y_test_pad, role_test_pad, x_test_dialogue_len)
  dev = (X_dev, y_dev_pad, role_dev_pad, x_dev_dialogue_len)
  model.train(train, dev)

if __name__ == '__main__':
  simple_classification_non_mutual()


