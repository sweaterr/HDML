# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from official.utils.export import export
from preprocessing import imagenet_preprocessing


def image_bytes_serving_input_fn(image_shape, dtype=tf.float32):
  """Serving input fn for raw jpeg images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    # Bounding box around the whole image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=dtype, shape=[1, 1, 4])
    height, width, num_channels = image_shape
    image, _ = imagenet_preprocessing.preprocess_image(
      image_bytes, bbox, height, width, num_channels, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
    shape=[None], dtype=tf.string, name='input_tensor')
  images = tf.map_fn(
    _preprocess_image, image_bytes_list, back_prop=False, dtype=dtype)
  return tf.estimator.export.TensorServingInputReceiver(
    images, {'image_bytes': image_bytes_list})


def export_pb(flags_core, flags_obj, shape, classifier):
  export_dtype = flags_core.get_tf_dtype(flags_obj)

  if not flags_obj.data_format:
    raise ValueError('The `data_format` must be specified: channels_first or channels_last ')

  bin_export_path = os.path.join(flags_obj.export_dir, flags_obj.data_format, 'binary_input')
  bin_input_receiver_fn = functools.partial(
    image_bytes_serving_input_fn, shape, dtype=export_dtype)

  pp_export_path = os.path.join(flags_obj.export_dir, flags_obj.data_format, 'preprocessed_input')
  pp_input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
    shape, batch_size=None, dtype=export_dtype)

  classifier.export_savedmodel(bin_export_path, bin_input_receiver_fn)
  classifier.export_savedmodel(pp_export_path, pp_input_receiver_fn)
