from data_provider import dialogue_dataset
from build_data import Config
import os
import pickle
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from model.dialouge_tagging_model import DialougeTagging


DATASET_NAME = 'Gotit'
PRETRAIN_DIR = './pre_train'
EMBEDDING_FILE = '/media/strnam/New Volume/competition/Toxic_comment/fasttext/crawl-300d-2M.vec'
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

  # tokenizer = Tokenizer()
  # tokenizer.fit_on_texts(x_train_flat)
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


  y_test_vec = to_categorical_non_mutual(y_test, num_tags)
  y_test_vec_flat = np.concatenate(y_test_vec, axis=0).tolist()
  y_test_pad = pad_sequences(y_test_vec, max_utt_in_sess, padding='post')
  role_test_pad = pad_sequences(role_test, maxlen=max_utt_in_sess, padding='post', value=-1)

  num_vocabs = 10000


  # ====
  model = DialougeTagging()
  model.max_word_in_utterance = max_length
  model.max_utterance_in_session = max(x_train_dialogue_len)
  model.num_classes = dataset.get_num_tags()
  model.evaluate_every = 5
  model.num_epochs = 50
  model.vocab_size = len(tokenizer.word_index)  # 86934 # vocab_size
  model.embedding_size = 100  # EMBEDDING_SIZE
  # model.pretrained_word_matrix = embedding_matrix
  model.pretrained_word_matrix = None

  model.utterance_filter_sizes = [3]
  model.utterance_num_filter = 2
  model.session_filter_sizes = [2]
  model.session_num_filters = 2


  checkpoint_dir = '/media/strnam/New Volume/DialoguageTagging/ModelRefactor/runs/1532963925/summaries/train/model.ckpt-30'
  model.restore_session(checkpoint_dir)
  raw_preds  = model.predict(X_test, role_test_pad, np.array(x_test_dialogue_len))
  preds = raw_preds[0]

  threshold = 0.5
  print(preds)
  preds[preds >= threshold] = 1
  preds[preds < threshold] = 0
  #print
  print(preds)
  print(np.array(y_test_vec_flat))

  pred_ids = np.argmax(preds, axis=1)
  df = dataset.set_predict_tag(dataset.test_set_idx, pred_ids)
  df[['Speaker', 'Content', 'predicts', 'SelectedOptionId']].to_csv('predict_gotit.csv', index=False)

  score = metric_absolute_match(np.array(y_test_vec_flat), preds)
  f1_s = f1(np.array(y_test_vec_flat), preds)
  f1_detail = detail_f1_score(np.array(y_test_vec_flat), preds)
  acc_detail = detail_accuracy_score(np.array(y_test_vec_flat), preds)
  precision_detail = detail_precision(np.array(y_test_vec_flat), preds)
  recall_detail = detail_recall(np.array(y_test_vec_flat), preds)

  print("Accuracy: ", score)
  print("F1: ", f1_s)
  print("Just match", metric_just_match(np.array(y_test_vec_flat), preds))
  print("F1 detail", f1_detail)
  print("Accuracy detail", acc_detail)
  print("Precision detail", precision_detail)
  print("Recall detail", recall_detail)


if __name__ == '__main__':
  simple_classification_non_mutual()


