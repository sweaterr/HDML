#!/usr/bin/env python
# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six
import tensorflow as tf

from preprocessing import cub_preprocessing
from preprocessing import vgg_preprocessing
from preprocessing import imagenet_preprocessing


def int64_feature(values):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(values, six.text_type):
    values = six.binary_type(values, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Wrapper for inserting floats features into Example proto."""
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def convert_to_example(image_data, image_format, class_id, height, width, bbox=None):
  """
  Build an Example proto for an example.
  :param image_data: JPEG encoding of RGB image
  :param image_format: Image format. 'jpg', 'gif' etc
  :param class_id: Integer, identifier for the ground truth for the network
  :param height: Image height in pixels
  :param width: Image width in pixels
  :param bbox: list of bounding boxes. each box is a list of integers specifying [xmin, ymin, xmax, ymax].
  :return: Example proto.
  """
  assert height > 0
  assert width > 0
  (xmin, ymin, xmax, ymax) = ([], [], [], [])
  bbox = [] if bbox is None else bbox
  for b in bbox:
    assert len(b) == 4
    [l.append(p) for l, p in zip([xmin, ymin, xmax, ymax], b)]

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': bytes_feature(image_data),
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/class/label': int64_feature(class_id),
    'image/format': bytes_feature(image_format),
    'image/object/bbox/xmin': float_feature(xmin),
    'image/object/bbox/xmax': float_feature(xmax),
    'image/object/bbox/ymin': float_feature(ymin),
    'image/object/bbox/ymax': float_feature(ymax)
  }))
  return example


def convert_to_example_without_bbox(image_data, image_format, class_id, height, width):
  """
  Build an Example proto for an example. Don't use it in combination with `convert_to_example`
  :param image_data: JPEG encoding of RGB image
  :param image_format: Image format. 'jpg', 'gif' etc
  :param class_id: Integer, identifier for the ground truth for the network
  :param height: Image height in pixels
  :param width: Image width in pixels
  :return: Example proto.
  """
  assert height > 0
  assert width > 0
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': bytes_feature(image_data),
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/class/label': int64_feature(class_id),
    'image/format': bytes_feature(image_format),
  }))
  return example


def mixup(x, y, alpha=0.2, keep_batch_size=True):
  dist = tf.contrib.distributions.Beta(alpha, alpha)

  _, h, w, c = x.get_shape().as_list()

  batch_size = tf.shape(x)[0]
  num_class = y.get_shape().as_list()[1]

  lam1 = dist.sample([batch_size // 2])

  if x.dtype == tf.float16:
    lam1 = tf.cast(lam1, dtype=tf.float16)
    y = tf.cast(y, dtype=tf.float16)

  x1, x2 = tf.split(x, 2, axis=0)
  y1, y2 = tf.split(y, 2, axis=0)

  lam1_x = tf.tile(tf.reshape(lam1, [batch_size // 2, 1, 1, 1]), [1, h, w, c])
  lam1_y = tf.tile(tf.reshape(lam1, [batch_size // 2, 1]), [1, num_class])

  mixed_sx1 = lam1_x * x1 + (1. - lam1_x) * x2
  mixed_sy1 = lam1_y * y1 + (1. - lam1_y) * y2
  mixed_sx1 = tf.stop_gradient(mixed_sx1)
  mixed_sy1 = tf.stop_gradient(mixed_sy1)

  if keep_batch_size:
    lam2 = dist.sample([batch_size // 2])

    if x.dtype == tf.float16:
      lam2 = tf.cast(lam2, dtype=tf.float16)

    lam2_x = tf.tile(tf.reshape(lam2, [batch_size // 2, 1, 1, 1]), [1, h, w, c])
    lam2_y = tf.tile(tf.reshape(lam2, [batch_size // 2, 1]), [1, num_class])

    x3 = tf.reverse(x2, [0])
    y3 = tf.reverse(y2, [0])

    mixed_sx2 = lam2_x * x1 + (1. - lam2_x) * x3
    mixed_sy2 = lam2_y * y1 + (1. - lam2_y) * y3

    mixed_sx2 = tf.stop_gradient(mixed_sx2)
    mixed_sy2 = tf.stop_gradient(mixed_sy2)

    mixed_sx1 = tf.concat([mixed_sx1, mixed_sx2], axis=0)
    mixed_sy1 = tf.concat([mixed_sy1, mixed_sy2], axis=0)

  return mixed_sx1, mixed_sy1


def get_filenames(is_training, data_dir, train_regex='train-*-of-*', val_regex='validation-*-of-*'):
  """Return filenames for dataset."""
  if is_training:
    path = os.path.join(data_dir, train_regex)
    matching_files = tf.gfile.Glob(path)
    return matching_files
  else:
    path = os.path.join(data_dir, val_regex)
    matching_files = tf.gfile.Glob(path)
    return matching_files


def parse_example_proto(example_serialized):
  """
  Parses an Example proto containing a training example of an image.
  The output of the build_xxx.py image preprocessing script is a dataset containing serialized Example pb.
  Each Example proto contains the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/class/label: 615
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/encoded: <JPEG encoded string>

    <<already defined but no implements.>>
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'

  :param example_serialized: scalar Tensor tf.string containing a serialized Example protocol buffer.
  :return:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
    {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                 'image/object/bbox/ymin',
                                 'image/object/bbox/xmax',
                                 'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, num_channels, dtype, use_random_crop=True, image_size=224,
                 autoaugment_type=None, with_drawing_bbox=False, dct_method="", preprocessing_type='imagenet'):
  """
  Parses a record containing a training example of an image.
  The input record is parsed into a label and image, and the image is passed through preprocessing steps.
  (cropping, flipping, and so on).
  :param raw_record: Scalar Tensor tf.string containing a serialized Example protocol buffer.
  :param is_training: A boolean denoting whether the input is for training.
  :param num_channels: The number of channels.
  :param dtype: Data type to use for images/features.
  :param use_random_crop:  Whether to randomly crop a training image.
  :param image_size: Output image size.
  :param autoaugment_type: Auto augmentation type.
  :param with_drawing_bbox: If True, return processed image tensor including raw image tensor with bbox.
  :param dct_method:  An optional `string`. Defaults to `""`.
    string specifying a hint about the algorithm used for decompression.
    Defaults to "" which maps to a system-specific default.
    Currently valid values are ["INTEGER_FAST", "INTEGER_ACCURATE"].
    The hint may be ignored.
    (e.g., the internal jpeg library changes to a version that does not have that specific option.)
  :param preprocessing_type: image preprocessing type. ['imagenet', 'cub']

  :return: Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = parse_example_proto(raw_record)
  raw_image_with_bbox = None
  print('preprocessing_type', preprocessing_type)
  if preprocessing_type == 'imagenet':
    image, raw_image_with_bbox = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=image_size,
      output_width=image_size,
      num_channels=num_channels,
      is_training=is_training,
      use_random_crop=use_random_crop,
      dct_method=dct_method,
      autoaugment_type=autoaugment_type,
      with_drawing_bbox=with_drawing_bbox)
  elif preprocessing_type == 'cub':
    image = cub_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      output_height=image_size,
      output_width=image_size,
      num_channels=num_channels,
      is_training=is_training,
      dct_method=dct_method,
      autoaugment_type=autoaugment_type)
  elif preprocessing_type == 'vgg':
    image = vgg_preprocessing.preprocess_image(
      image=image_buffer,
      output_height=image_size,
      output_width=image_size,
      is_training=is_training,
      autoaugment_type=autoaugment_type)
  else:
    raise NotImplementedError

  image = tf.cast(image, dtype)
  if with_drawing_bbox and not raw_image_with_bbox:
    image = tf.stack([image, raw_image_with_bbox])

  return image, label


def process_record_dataset_ir(dataset, is_training, batch_size, parse_record_fn, num_classes, num_channels, num_gpus=0,
                              use_random_crop=True, dtype=tf.float32, with_drawing_bbox=False, autoaugment_type=None,
                              num_instances=2, is_aggregated=False, preprocessing_type='imagenet', dct_method=""):
  """
  Given a Dataset with raw records, return an iterator over the records.
  :param dataset: A Dataset representing raw records
  :param is_training: A boolean denoting whether the input is for training.
  :param batch_size: The number of samples per batch. It must be multiple of two.
  :param parse_record_fn: A function that takes a raw record and returns the corresponding (image, label) pair.
  :param num_classes: 
  :param num_channels: 
  :param num_gpus: The number of gpus used for training.
  :param use_random_crop: Whether to randomly crop a training image.
  :param dtype: Data type to use for images/features.
  :param with_drawing_bbox: If True, return the dataset including raw image tensor with bbox.
  :param autoaugment_type: Auto augmentation type. 'imagenet', 'svhn', 'cifar', `good`
  :param num_instances: The number of instances.
  :param is_aggregated: 
  :param preprocessing_type: 
  :param dct_method:  An optional `string`. Defaults to `""`.
    string specifying a hint about the algorithm used for decompression.
    Defaults to "" which maps to a system-specific default.
    Currently valid values are ["INTEGER_FAST", "INTEGER_ACCURATE"].
    The hint may be ignored.
    (e.g., the internal jpeg library changes to a version that does not have that specific option.)
  :return: Dataset of (image, label) pairs ready for iteration.
  """

  def _choose_random_labels_from_pairwises(x):
    assert batch_size % num_instances == 0
    x = tf.transpose(x)
    x = tf.random_shuffle(x, seed=0)

    return tf.transpose(x[:int(batch_size / num_instances)])

  def _parse_records(values):
    return tf.map_fn(lambda raw_record: parse_record_fn(raw_record,
                                                        is_training,
                                                        num_channels,
                                                        dtype,
                                                        dct_method=dct_method,
                                                        use_random_crop=use_random_crop,
                                                        autoaugment_type=autoaugment_type,
                                                        preprocessing_type=preprocessing_type,
                                                        with_drawing_bbox=with_drawing_bbox),
                     values,
                     dtype=(dtype, tf.int32))

  def _get_features_and_labels(values):
    features = []
    labels = []
    for i in range(num_instances):
      parsed_a_instance = _parse_records(values[i])
      features.append(parsed_a_instance[0])
      labels.append(parsed_a_instance[1])

    if is_aggregated:
      h = features[0].get_shape().as_list()[1]
      w = features[0].get_shape().as_list()[2]
      features = tf.reshape(features, [-1, h, w, num_channels])
      labels = tf.reshape(labels, [-1, ])
    return features, labels

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through processing.
  dataset = dataset.prefetch(buffer_size=num_classes * num_instances)

  # Grouping anchor records and positive records. Each records consists of a image by class.
  dataset = dataset.batch(num_classes)
  dataset = dataset.batch(num_instances)

  dataset = dataset.map(lambda x: _choose_random_labels_from_pairwises(x))
  dataset = dataset.map(lambda x: _get_features_and_labels(x))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, num_channels, parse_record_fn, num_epochs,
                           num_gpus, examples_per_epoch, dtype, use_random_crop=False, with_drawing_bbox=False,
                           autoaugment_type=None, dct_method='', preprocessing_type='imagenet',drop_remainder=False):
  """
  Given a Dataset with raw records, return an iterator over the records.

  :param dataset: A Dataset representing raw records
  :param is_training: A boolean denoting whether the input is for training.
  :param batch_size: The number of samples per batch.
  :param shuffle_buffer: The buffer size to use when shuffling records.
    A larger value results in better randomness, but smaller values reduce startup time and use less memory.
  :param num_channels: The number of channels.
  :param parse_record_fn: A function that takes a raw record and returns the corresponding (image, label) pair.
  :param num_epochs: The number of epochs to repeat the dataset.
  :param num_gpus: The number of gpus used for training.
  :param examples_per_epoch: The number of examples in an epoch.
  :param dtype: Data type to use for images/features.
  :param use_random_crop: Whether to randomly crop a training image.
  :param with_drawing_bbox: If True, return image including raw image tensor with bbox.
  :param autoaugment_type: Auto augmentation type.
  :param dct_method:  An optional `string`. Defaults to `""`.
    string specifying a hint about the algorithm used for decompression.
    Defaults to "" which maps to a system-specific default.
    Currently valid values are ["INTEGER_FAST", "INTEGER_ACCURATE"].
    The hint may be ignored.
    (e.g., the internal jpeg library changes to a version that does not have that specific option.)
  :param preprocessing_type: 
  :return: Dataset of (image, label) pairs ready for iteration.
  """
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)

  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
    tf.contrib.data.map_and_batch(
        lambda value: parse_record_fn(value,
                                      is_training,
                                      num_channels,
                                      dtype,
                                      dct_method=dct_method,
                                      use_random_crop=use_random_crop,
                                      autoaugment_type=autoaugment_type,
                                      preprocessing_type=preprocessing_type,
                                      with_drawing_bbox=with_drawing_bbox),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=drop_remainder))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset
