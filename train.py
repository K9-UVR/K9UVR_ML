from absl import app, flags, logging
import tensorflow as tf
import os
from pathlib import Path
import IPython.display as display
import numpy as np


FLAGS = flags.FLAGS
# flags.DEFINE_string("name", None, "Your name.")
# flags.DEFINE_integer("num_times", 1,
#                      "Number of times to print greeting.")

flags.DEFINE_string("dataset_folder", None, "Path to tfrecords")

# # Required flag.
flags.mark_flag_as_required("dataset_folder")

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def main(argv):
    del argv  # Unused.
    logging.set_verbosity(logging.INFO)
    logging.info("Model training started")

    # create dataset from tfrecords files
    dataset = createDataset(FLAGS.dataset_folder)

    # parsed_image_dataset = dataset.take(2).map(_parse_image_function)

    # dataset = tf.data.TFRecordDataset("/Users/konradkarimi/Projects/K9_YOLO/Data/CXX_OUT/K9_CXX-TFRecords-export/18422075_1133923726719385_5454745554831529_o.tfrecord")

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def createDataset(tfrecords_folder):
    # data_dir = Path(tfrecords_folder)
    # files = data_dir.glob('*.tfrecord')
    # files_str = ','.join(map(str, files))
    # dataset = tf.data.TFRecordDataset(files)
    files = tf.data.Dataset.list_files(tfrecords_folder + "*.tfrecord")
    dataset = tf.data.TFRecordDataset(files)
    return dataset


if __name__ == '__main__':
    app.run(main)
