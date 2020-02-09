from tqdm import tqdm
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from six import BytesIO
from pprint import pprint
import os
from absl import app, flags, logging
import tensorflow_hub as hub
import util

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# /Users/konradkarimi/Downloads/doggo.jpg

test_img = "/Users/konradkarimi/Downloads/doggo.jpg"

FLAGS = flags.FLAGS
# flags.DEFINE_integer("num_times", 1, "Number of times to print greeting.")
# flags.DEFINE_string('name', None, 'Your name.')
# flags.DEFINE_integer('age', 25, 'Your age in years.', lower_bound=0)
# flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
# flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')

# Required flag.
# flags.mark_flag_as_required("name")

flags.DEFINE_string("img_addr", None, "addr to test img")
flags.mark_flag_as_required("img_addr")


def run_interference(img_addr):
    with tf.Graph().as_default():
        image_sting_placeholder = tf.compat.v1.placeholder(tf.string)
        decoded_image = tf.image.decode_jpeg(image_sting_placeholder)
        decoded_image_float = tf.image.convert_image_dtype(
            image=decoded_image, dtype=tf.float32)

        image_tensor = tf.expand_dims(decoded_image_float, 0)

        model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        detector = hub.Module(model_url)
        detector_output = detector(image_tensor, as_dict=True)

        init_ops = [tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.tables_initializer()]
        sess = tf.compat.v1.Session()
        sess.run(init_ops)

    with tf.compat.v1.gfile.Open(img_addr, "rb") as binfile:
        imagestring = binfile.read()

    result_out, image_out = sess.run([detector_output, decoded_image], feed_dict={
                                     image_sting_placeholder: imagestring})

    image_with_boxes = util.draw_boxes(np.array(
        image_out), result_out["detection_boxes"], result_out["detection_class_entities"], result_out["detection_scores"])
    util.display_image(image_with_boxes)


def main(argv):
    logging.set_verbosity(logging.INFO)
    del argv  # Unused.
    run_interference(FLAGS.img_addr)


if __name__ == '__main__':
    app.run(main)
