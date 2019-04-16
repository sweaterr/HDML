from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from functions import data_config
from utils import data_util


def input_fn(is_training,
             filenames,
             use_random_crop,
             batch_size,
             num_train_files,
             num_images,
             shuffle_buffer,
             num_channels,
             num_epochs=1,
             num_gpus=None,
             dtype=tf.float32,
             autoaugment_type=None,
             with_drawing_bbox=False,
             preprocessing_type='imagenet',
             drop_remainder=False,
             dct_method=""):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    use_random_crop: Whether to randomly crop a training image.
    batch_size: The number of samples per batch.
    num_train_files: The number of train files.
    num_images: The number of images.
    shuffle_buffer: The size of file shuffle buffer.
    num_channels: The number of channels.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features
    dct_method: An optional `string`. Defaults to `""`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to "" which maps to a system-specific
      default.  Currently valid values are ["INTEGER_FAST",
      "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    autoaugment_type: Auto augmentation type. 'imagenet', 'svhn', 'cifar'
    with_drawing_bbox: If True, return the dataset including raw image tensor with bbox.

  Returns:
    A dataset that can be used for iteration.
  """
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=num_train_files)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  dataset = dataset.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=20))

  return data_util.process_record_dataset(
    dataset=dataset,
    is_training=is_training,
    batch_size=batch_size,
    shuffle_buffer=shuffle_buffer,
    parse_record_fn=data_util.parse_record,
    num_epochs=num_epochs,
    num_gpus=num_gpus,
    num_channels=num_channels,
    examples_per_epoch=num_images if is_training else None,
    dtype=dtype,
    use_random_crop=use_random_crop,
    dct_method=dct_method,
    autoaugment_type=autoaugment_type,
    preprocessing_type=preprocessing_type,
    drop_remainder=drop_remainder,
    with_drawing_bbox=with_drawing_bbox)


def input_fn_cls(is_training,
                 use_random_crop,
                 data_dir,
                 batch_size,
                 num_epochs=1,
                 num_gpus=None,
                 dtype=tf.float32,
                 with_drawing_bbox=False,
                 autoaugment_type=None,
                 dataset_name=None,
                 drop_remainder=False,
                 dct_method=""):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    use_random_crop: Whether to randomly crop a training image.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features
    autoaugment_type: Auto augmentation type. 'imagenet', 'svhn', 'cifar', 'good'
    with_drawing_bbox: If True, return the dataset including raw image tensor with bbox.
    dct_method: An optional `string`. Defaults to `""`.
    string specifying a hint about the algorithm used for
    decompression.  Defaults to "" which maps to a system-specific
    default.  Currently valid values are ["INTEGER_FAST",
    "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
    jpeg library changes to a version that does not have that specific
    option.)

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = data_util.get_filenames(is_training, data_dir)
  dataset = data_config.get_config(dataset_name)

  return input_fn(is_training, filenames, use_random_crop, batch_size,
                  dataset.num_train_files, dataset.num_images['train'], dataset.shuffle_buffer,
                  dataset.num_channels, num_epochs, num_gpus, dtype,
                  autoaugment_type=autoaugment_type,
                  with_drawing_bbox=with_drawing_bbox,
                  drop_remainder=drop_remainder,
                  dct_method=dct_method)


def input_fn_bfe_train(is_training,
                       use_random_crop,
                       data_dir,
                       batch_size,
                       train_epochs,
                       num_gpus=0,
                       dtype=tf.float32,
                       with_drawing_bbox=False,
                       autoaugment_type=None,
                       dct_method="",
                       preprocessing_type='imagenet',
                       num_instances=4,
                       dataset_name=None):
  """Input function which provides batches for train or eval.

  See https://oss.navercorp.com/VisualSearch/food-fighters/pull/49#issue-566301.

  Args:
    is_training: A boolean denoting whether the input is for training.
    use_random_crop: Whether to randomly crop a training image.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    train_epochs : The number of steps to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features
    with_drawing_bbox: If True, return the dataset including raw image tensor with bbox.
    autoaugment_type: Auto augmentation type. 'imagenet', 'svhn', 'cifar', `good`
    dct_method: An optional `string`. Defaults to `""`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to "" which maps to a system-specific
      default.  Currently valid values are ["INTEGER_FAST",
      "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    preprocessing_type: TODO
    num_instances: TODO
    dataset_name: TODO

  Returns:
    A dataset that can be used for iteration.
  """
  dataset_config = data_config.get_config(dataset_name)
  all_choices_ds = []
  for filename in data_util.get_filenames(True, data_dir, train_regex='train-label*'):
    dataset = tf.data.Dataset.from_tensors(filename)

    # The cycle_length isn't necessary now because there's only one tfrecord file.
    # Use this feature when you want to increase the number of file shards.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=1))
    # shuffling records by class. A larger shuffle buffer's size results in better randomness,
    # but smaller size reduce startup time and use less memory.
    dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(buffer_size=100)  # TODO adaptive buffer size
    )

    all_choices_ds.append(dataset)
  # A Repeat number must be mutliples of two which means anchor and positive.
  max_train_steps = train_epochs * int(dataset_config.num_images['train'] / batch_size)
  choice_dataset = tf.data.Dataset.range(len(all_choices_ds)).repeat(max_train_steps * num_instances)

  dataset = tf.contrib.data.choose_from_datasets(all_choices_ds, choice_dataset)

  return data_util.process_record_dataset_ir(
    dataset=dataset,
    is_training=is_training,
    batch_size=batch_size,
    parse_record_fn=data_util.parse_record,
    num_classes=dataset_config.num_classes,
    num_channels=dataset_config.num_channels,
    num_gpus=num_gpus,
    use_random_crop=use_random_crop,
    dtype=dtype,
    with_drawing_bbox=with_drawing_bbox,
    autoaugment_type=autoaugment_type,
    num_instances=num_instances,
    is_aggregated=True,
    preprocessing_type=preprocessing_type,
    dct_method=dct_method)


def input_fn_bfe_eval(is_training,
                      data_dir,
                      batch_size,
                      num_epochs=1,
                      num_gpus=0,
                      dtype=tf.float32,
                      preprocessing_type='imagenet',
                      dataset_name=None,
                      dct_method=""):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features
    preprocessing_type: TODO
    dataset_name: TODO
    dct_method: An optional `string`. Defaults to `""`.
    string specifying a hint about the algorithm used for
    decompression.  Defaults to "" which maps to a system-specific
    default.  Currently valid values are ["INTEGER_FAST",
    "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
    jpeg library changes to a version that does not have that specific
    option.)

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = data_util.get_filenames(is_training, data_dir, val_regex='validation-label*')
  dataset_config = data_config.get_config(dataset_name)

  return input_fn(is_training, filenames, False, batch_size,
                  dataset_config.num_train_files, dataset_config.num_images['validation'],
                  dataset_config.shuffle_buffer, dataset_config.num_channels, num_epochs, num_gpus, dtype,
                  preprocessing_type=preprocessing_type, dct_method=dct_method)


def input_fn_npair_train(is_training,
                         use_random_crop,
                         data_dir,
                         batch_size,
                         train_epochs,
                         num_gpus=0,
                         dtype=tf.float32,
                         with_drawing_bbox=False,
                         autoaugment_type=None,
                         preprocessing_type='imagenet',
                         dct_method="",
                         dataset_name=None):
  """Input function which provides batches for train or eval.
  See https://oss.navercorp.com/VisualSearch/food-fighters/pull/49#issue-566301.

  Args:
    is_training: A boolean denoting whether the input is for training.
    use_random_crop: Whether to randomly crop a training image.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    train_epochs: TODO
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features
    dct_method: An optional `string`. Defaults to `""`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to "" which maps to a system-specific
      default.  Currently valid values are ["INTEGER_FAST",
      "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    autoaugment_type: Auto augmentation type. 'imagenet', 'svhn', 'cifar', `good`
    with_drawing_bbox: If True, return the dataset including raw image tensor with bbox.


  Returns:
    A dataset that can be used for iteration.
  """
  dconf = data_config.get_config(dataset_name)
  all_choices_ds = []
  for filename in data_util.get_filenames(True, data_dir, train_regex='train-label*'):
    dataset = tf.data.Dataset.from_tensors(filename)

    # The cycle_length isn't necessary now because there's only one tfrecord file.
    # Use this feature when you want to increase the number of file shards.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=1))
    # shuffling records by class. A larger shuffle buffer's size results in better randomness,
    # but smaller size reduce startup time and use less memory.
    dataset = dataset.shuffle(buffer_size=100, seed=0)
    dataset = dataset.repeat()

    all_choices_ds.append(dataset)
  # A Repeat number must be mutliples of two which means anchor and positive.
  max_train_steps = train_epochs * int(dconf.num_images['train'] / batch_size)
  choice_dataset = tf.data.Dataset.range(len(all_choices_ds)).repeat(max_train_steps * 2)

  dataset = tf.contrib.data.choose_from_datasets(all_choices_ds, choice_dataset)

  return data_util.process_record_dataset_ir(
    dataset=dataset,
    is_training=is_training,
    batch_size=batch_size,
    parse_record_fn=data_util.parse_record,
    num_classes=dconf.num_classes,
    num_channels=dconf.num_channels,
    num_gpus=num_gpus,
    use_random_crop=use_random_crop,
    dtype=dtype,
    with_drawing_bbox=with_drawing_bbox,
    autoaugment_type=autoaugment_type,
    num_instances=2,
    preprocessing_type=preprocessing_type,
    is_aggregated=False,
    dct_method=dct_method)
