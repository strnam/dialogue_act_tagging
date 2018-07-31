import json
import os
import numpy as np
import pandas as pd
import re
from .SwDA_config import SwDA_config
from .Maluuba_config import Maluuba_config
from .GotIt_config import GotIt_config

SWITCHBOARD_DATA_PATH = '../data/SwDA.csv'
MALUUBA_DATA_PATH = '../data/maluuba.csv'
# GOTIT_DATA_PATH = '../data/gotIt_dialogue.csv'
# GOTIT_DATA_PATH = '../data/final_cleandata20183005.csv'
GOTIT_DATA_PATH = '../data/final_cleandata20183005.csv'



def get_dataset(dataset_name):
  if dataset_name == "Switchboard":
    return SwDA()
  if dataset_name == "Maluuba":
    return MaluubaDataset()
  if dataset_name == 'Gotit':
    return GotItDataset()

# def clean_str(string):
#   """
#   Tokenization/string cleaning for all datasets except for SST.
#   Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#   """
#   string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#   string = re.sub(r"\'s", " \'s", string)
#   string = re.sub(r"\'ve", " \'ve", string)
#   string = re.sub(r"n\'t", " n\'t", string)
#   string = re.sub(r"\'re", " \'re", string)
#   string = re.sub(r"\'d", " \'d", string)
#   string = re.sub(r"\'ll", " \'ll", string)
#   string = re.sub(r",", " , ", string)
#   string = re.sub(r"!", " ! ", string)
#   string = re.sub(r"\(", " \( ", string)
#   string = re.sub(r"\)", " \) ", string)
#   string = re.sub(r"\?", " \? ", string)
#   string = re.sub(r"\s{2,}", " ", string)
#   return string.strip().lower()

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  string = re.sub(r"(\bhttp\S+|\S+\.\S+)", 'urllabel', string)
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  # string = re.sub(r'((^| )\d+( |$))', " numberlabel ", string)
  string = re.sub(r'(?=\b)\d+(?<=\b)', " numberlabel ", string)
  string = re.sub(r'(?=\b)\d+\w+(?<=\b)', "numberletter", string)
  string = re.sub(r'(?=\b)\w+\d+(?<=\b)', "letternumber", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()

class DialogueDataset(object):
  def __init__(self):
    raise NotImplemented

  def split_data_to_train(self):
    raise NotImplemented

  def get_dialogue_data(self, dialogue_ids):
    raise NotImplemented

  def flat_dialogue(self, dialogues):
    utterances = []
    for utts_dialogue in dialogues:
      utterances.append( utts_dialogue)

    return np.concatenate(utterances, axis=0)

  def get_dialogues_length(self, dialogues):
    len_dialogues = [len(dialogue) for dialogue in dialogues]
    return len_dialogues

  def group_utterance_to_dialogue(self, utts, len_dialogues):
    assert(len(utts), np.sum(len_dialogues))
    cur_pos = 0
    l_dialogue = []
    for len_d in len_dialogues:
      dialouge = utts[cur_pos:(cur_pos+len_d)]
      cur_pos += len_d
      l_dialogue.append(dialouge)
    return l_dialogue

class GotItDataset(DialogueDataset):
  def __init__(self, data_path=None):
    if data_path is None:
      dir_path = os.path.dirname(os.path.realpath(__file__))
      self.df = pd.read_csv(os.path.join(dir_path, GOTIT_DATA_PATH))
      self.df['SelectedOptionId'] = self.df['SelectedOptionId'] - 1

    # self.act_id_map = Maluuba_config.maluuba_act_id_map
    self.test_set_idx = GotIt_config.test_set_idx
    self.train_set_idx = GotIt_config.train_set_idx
    # self.train_set_idx = list(set(self.df['SessionId']) - set(self.test_set_idx))
    self.valid_set_idx = GotIt_config.valid_set_idx
    self.act_id_map = GotIt_config.act_id_map


  def split_data_to_train(self):
    return self.train_set_idx, self.valid_set_idx, self.test_set_idx

  def get_dialogue_data(self, dialogue_ids):
    l_dialogue_text = []
    l_dialogue_act_tags = []
    for dialogue_id in dialogue_ids:
      dialogue_text = self.df[self.df['SessionId'] == dialogue_id]['Content'].tolist()
      dialogue_act_tags = self.df[self.df['SessionId'] == dialogue_id]['SelectedOptionId'].tolist()
      # dialogue_clean = [clean_str(utt) for utt in  dialogue_text]

      l_dialogue_text.append(dialogue_text)
      # l_dialogue_text.append(dialogue_clean)
      l_dialogue_act_tags.append(dialogue_act_tags)
    return l_dialogue_text, l_dialogue_act_tags

  def get_role_data(self, dialogue_ids):
    l_role_id = []
    for dialogue_id in dialogue_ids:
      role_text = self.df[self.df['SessionId'] == dialogue_id]['Speaker'].tolist()
      role_ids = [GotIt_config.gotit_role_dict[role] for role in role_text]
      l_role_id.append(role_ids)

    return l_role_id


  def get_num_tags(self):
    return len(self.df['SelectedOptionId'].unique())

  def set_predict_tag(self, dialogue_ids, predicts):
    l_df = []
    for dialogue_id in dialogue_ids:
      dialogue_text = self.df[self.df['SessionId'] == dialogue_id]
      l_df.append(dialogue_text)

    df_predict = pd.concat(l_df)
    df_predict['predicts'] = predicts
    return df_predict


class MaluubaDataset(DialogueDataset):
  def __init__(self):
    # columns: act_tag, caller,	clean_text,	conversation_no,	damsl_act_tag,	act_tag_id
    dir_path = os.path.dirname(os.path.realpath(__file__))
    self.df = pd.read_csv(os.path.join(dir_path, MALUUBA_DATA_PATH))
    self.df['act_ids'] = self.df['act_ids'].apply(lambda row: json.loads(row))

    self.act_id_map = Maluuba_config.maluuba_act_id_map
    self.train_set_idx = list(range(1, 1001))
    self.test_set_idx = list(range(1001, 1369+1))
    self.valid_set_idx = []

  def split_data_to_train(self):
    return self.train_set_idx, self.valid_set_idx, self.test_set_idx


  def get_dialogue_data(self, dialogue_ids):
    l_dialogue_text = []
    l_dialogue_act_tags = []
    for dialogue_id in dialogue_ids:
      dialogue_text = self.df[self.df['session_num_id'] == dialogue_id]['text'].tolist()
      dialogue_act_tags = self.df[self.df['session_num_id'] == dialogue_id]['act_ids'].tolist()
      #dialogue_clean = [clean_str(utt) for utt in  dialogue_text]

      l_dialogue_text.append(dialogue_text)
      l_dialogue_act_tags.append(dialogue_act_tags)
    return l_dialogue_text, l_dialogue_act_tags

  def get_role_data(self, dialogue_ids):
    l_role_id = []
    for dialogue_id in dialogue_ids:
      role_text = self.df[self.df['session_num_id'] == dialogue_id]['author'].tolist()
      role_ids = [Maluuba_config.maluuba_role_dict[role] for role in role_text]
      l_role_id.append(role_ids)

    return l_role_id


  def get_num_tags(self):
    return len(self.act_id_map)


class SwDA(DialogueDataset):
  def __init__(self):
    # columns: act_tag, caller,	clean_text,	conversation_no,	damsl_act_tag,	act_tag_id
    dir_path = os.path.dirname(os.path.realpath(__file__))
    self.df = pd.read_csv(os.path.join(dir_path, SWITCHBOARD_DATA_PATH))
    self.train_set_idx = [self.__parse_id(idx) for idx in SwDA_config.train_set_idx]
    self.test_set_idx = [self.__parse_id(idx) for idx in SwDA_config.test_set_idx]
    self.valid_set_idx = [self.__parse_id(idx) for idx in SwDA_config.valid_set_idx]

  def __parse_id(self, sw_session_string):
    return int(sw_session_string[2:])


  def split_data_to_train(self):
    return self.train_set_idx, self.valid_set_idx, self.test_set_idx

  def get_dialogue_data(self, dialogue_ids):
    l_dialogue_text = []
    l_dialogue_act_tags = []
    for dialogue_id in dialogue_ids:
      dialogue_text = self.df[self.df['conversation_no'] == dialogue_id]['clean_text'].tolist()
      dialogue_act_tags = self.df[self.df['conversation_no'] == dialogue_id]['act_tag_id'].tolist()
      dialogue_clean = [clean_str(utt) for utt in  dialogue_text]

      l_dialogue_text.append(dialogue_clean)
      l_dialogue_act_tags.append(dialogue_act_tags)
    return l_dialogue_text, l_dialogue_act_tags

if __name__ == "__main__":
  dataset = MaluubaDataset()
  train_ids, _, _ = dataset.split_data_to_train()
  x, y = dataset.get_dialogue_data(train_ids)
  x_flat = dataset.flat_dialogue(x)
  len_dialouge = dataset.get_dialogues_length(x)
  print(len(x))
  print(len(y[0]))
  print(len(x_flat))
  new_x = dataset.group_utterance_to_dialogue(x_flat, len_dialouge)
  print(len(new_x))
  new_len_dialouge = dataset.get_dialogues_length(new_x)
  print(sum(new_len_dialouge), sum(len_dialouge))