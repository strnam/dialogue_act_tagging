from data_provider import dialogue_dataset


class Config(object):
  def __init__(self, dataset_name, load=True):
    self.dataset_name = dataset_name
    if load:
      self.load_data()

  def load_data(self):
    # LOAD DATA
    dataset = dialogue_dataset.get_dataset(self.dataset_name)

    # SPLIT DATA TO TRAIN, VALID, TEST
    self.train_ids, self.valid_ids, self.test_ids = dataset.split_data_to_train()