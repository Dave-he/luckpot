from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

train_file_path = 'ball.csv'

with open(train_file_path, 'r') as f:
    names_row = f.readline()


CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)

LABEL_COLUMN = 'survived'

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)

examples, labels = next(iter(raw_train_data)) # Just the first batch.
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)