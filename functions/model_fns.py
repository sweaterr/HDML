from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from functions import data_config
from nets import resnet_model
from nets import run_loop_bfe
from nets import run_loop_classification
from nets import run_loop_npair


def keep_prob_decay(starter_kp, end_kp, decay_steps):
  def keep_prob_decay_fn(global_step):
    kp = tf.train.polynomial_decay(starter_kp, global_step, decay_steps,
                                   end_kp, power=1.0,
                                   cycle=False)
    return kp

  return keep_prob_decay_fn


def learning_rate_with_decay(learning_rate_decay_type,
                             batch_size, batch_denom, num_images, num_epochs_per_decay,
                             learning_rate_decay_factor, end_learning_rate, piecewise_lr_boundary_epochs,
                             piecewise_lr_decay_rates, base_lr, warmup=False, train_epochs=None):
  """Get a learning rate that decays step-wise as training progresses.
  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    num_epochs_per_decay: number of epochs after which learning rate decays.
    learning_rate_decay_factor: number of epochs after which learning rate decays.
    end_learning_rate: the minimal end learning rate used by a polynomial decay learning rate.
    piecewise_lr_boundary_epochs: A list of ints with strictly increasing entries to reduce the learning rate at certain epochs.
    piecewise_lr_decay_rates: A list of floats that specifies the decay rates for the intervals defined by piecewise_lr_boundary_epochs. It should have one more element than piecewise_lr_boundary_epochs.
    base_lr: initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
    train_epochs: The number of train epochs
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  tf.logging.debug("learning_rate_decay_type=({})".format(learning_rate_decay_type))
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size
  decay_steps = int(batches_per_epoch * num_epochs_per_decay)

  def learning_rate_fn(global_step):
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
    else:
      warmup_steps = 0

    if learning_rate_decay_type == 'exponential':
      lr = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps,
                                      learning_rate_decay_factor, staircase=True,
                                      name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
      lr = tf.constant(base_lr, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
      lr = tf.train.polynomial_decay(initial_learning_rate, global_step, decay_steps,
                                     end_learning_rate, power=1.0,
                                     cycle=False, name='polynomial_decay_learning_rate')
    elif learning_rate_decay_type == 'piecewise':
      boundaries = [int(batches_per_epoch * epoch) for epoch in piecewise_lr_boundary_epochs]
      vals = [initial_learning_rate * decay for decay in piecewise_lr_decay_rates]
      lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    elif learning_rate_decay_type == 'cosine':
      total_batches = int(batches_per_epoch * train_epochs) - warmup_steps
      global_step_except_warmup_step = global_step - warmup_steps
      lr = tf.train.cosine_decay(initial_learning_rate, global_step_except_warmup_step, total_batches)
    else:
      raise NotImplementedError

    if warmup:
      warmup_lr = (
              initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
        warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr

  return learning_rate_fn


def get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
      resnet_size, choices.keys()))
    raise ValueError(err)


class Model(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size,
               data_format=None,
               num_classes=None,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE,
               dim_features=64,
               no_downsample=False,
               zero_gamma=False,
               use_se_block=False):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
      dim_features: TODO
      no_downsample: TODO
      zero_gamma: we initialize gamma = 0 for all BN layers that sit at the end of a residual block.
      use_se_block: Use Squeeze Excitation block.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    if no_downsample:
      block_strides = [1, 2, 2, 1]
    else:
      block_strides = [1, 2, 2, 2]

    super(Model, self).__init__(
      resnet_size=resnet_size,
      bottleneck=bottleneck,
      num_classes=num_classes,
      num_feature=dim_features,
      num_filters=64,
      kernel_size=7,
      conv_stride=2,
      first_pool_size=3,
      first_pool_stride=2,
      block_sizes=get_block_sizes(resnet_size),
      block_strides=block_strides,
      resnet_version=resnet_version,
      data_format=data_format,
      dtype=dtype,
      zero_gamma=zero_gamma,
      use_se_block=use_se_block
    )


def model_fn_cls(features, labels, mode, params):
  if int(params['resnet_size']) < 50:
    assert not params['use_dropblock']
    assert not params['use_se_block']
    assert not params['use_resnet_d']

  dataset = data_config.get_config(params['dataset_name'])
  learning_rate_fn = learning_rate_with_decay(
    learning_rate_decay_type=params['learning_rate_decay_type'],
    batch_size=params['batch_size'], batch_denom=params['batch_size'],
    num_images=dataset.num_images['train'], num_epochs_per_decay=params['num_epochs_per_decay'],
    learning_rate_decay_factor=params['learning_rate_decay_factor'],
    end_learning_rate=params['end_learning_rate'],
    piecewise_lr_boundary_epochs=params['piecewise_lr_boundary_epochs'],
    piecewise_lr_decay_rates=params['piecewise_lr_decay_rates'],
    base_lr=params['base_learning_rate'],
    train_epochs=params['train_epochs'],
    warmup=params['lr_warmup'])

  if params['use_dropblock']:
    starter_kp = params['dropblock_kp'][0]
    end_kp = params['dropblock_kp'][1]
    batches_per_epoch = dataset.num_images['train'] / params['batch_size']
    decay_steps = int(params['train_epochs'] * batches_per_epoch)
    keep_prob_fn = keep_prob_decay(starter_kp, end_kp, decay_steps)
  else:
    keep_prob_fn = None

  return run_loop_classification.resnet_model_fn(
    features=features,
    labels=labels,
    num_classes=dataset.num_classes,
    mode=mode,
    model_class=Model,
    resnet_size=params['resnet_size'],
    weight_decay=params['weight_decay'],
    learning_rate_fn=learning_rate_fn,
    momentum=params['momentum'],
    zero_gamma=params['zero_gamma'],
    use_resnet_d=params['use_resnet_d'],
    label_smoothing=params['label_smoothing'],
    data_format=params['data_format'],
    resnet_version=params['resnet_version'],
    loss_scale=params['loss_scale'],
    loss_filter_fn=None,
    dtype=params['dtype'],
    fine_tune=params['fine_tune'],
    use_se_block=params['use_se_block'],
    display_raw_images_with_bbox=params['display_raw_images_with_bbox'],
    use_ranking_loss=params['use_ranking_loss'],
    mixup_type=params['mixup_type'],
    rollback_period=params['rollback_period'],
    rollback_lr_multiplier=params['rollback_lr_multiplier'],
    keep_prob_fn=keep_prob_fn)


def model_fn_bfe(features, labels, mode, params):
  assert int(params['resnet_size']) >= 50

  dataset = data_config.get_config(params['dataset_name'])
  learning_rate_fn = learning_rate_with_decay(
    learning_rate_decay_type=params['learning_rate_decay_type'],
    batch_size=params['batch_size'], batch_denom=params['batch_size'],
    num_images=dataset.num_images['train'], num_epochs_per_decay=params['num_epochs_per_decay'],
    learning_rate_decay_factor=params['learning_rate_decay_factor'],
    end_learning_rate=params['end_learning_rate'],
    piecewise_lr_boundary_epochs=params['piecewise_lr_boundary_epochs'],
    piecewise_lr_decay_rates=params['piecewise_lr_decay_rates'],
    base_lr=params['base_learning_rate'],
    train_epochs=params['train_epochs'],
    warmup=params['lr_warmup'])

  if params['use_dropblock']:
    starter_kp = params['dropblock_kp'][0]
    end_kp = params['dropblock_kp'][1]
    batches_per_epoch = dataset.num_images['train'] / params['batch_size']
    decay_steps = int(params['train_epochs'] * batches_per_epoch)
    keep_prob_fn = keep_prob_decay(starter_kp, end_kp, decay_steps)
  else:
    keep_prob_fn = None

  return run_loop_bfe.resnet_ir_model_fn(
    features=features,
    labels=labels,
    mode=mode,
    model_class=Model,
    resnet_size=params['resnet_size'],
    weight_decay=params['weight_decay'],
    learning_rate_fn=learning_rate_fn,
    momentum=params['momentum'],
    data_format=params['data_format'],
    dim_features=params['dim_features'],
    num_classes=dataset.num_classes,
    resnet_version=params['resnet_version'],
    loss_scale=params['loss_scale'],
    loss_filter_fn=None,
    dtype=params['dtype'],
    fine_tune=params['fine_tune'],
    zero_gamma=params['zero_gamma'],
    use_resnet_d=params['use_resnet_d'],
    no_downsample=params['no_downsample'],
    weight_softmax_loss=params['weight_softmax_loss'],
    weight_ranking_loss=params['weight_ranking_loss'],
    label_smoothing=params['label_smoothing'],
    use_bfe=params['use_bfe'],
    use_global_branch=params['use_global_branch'],
    ranking_loss_type=params['ranking_loss_type'],
    off_bfe_ranking=params['off_bfe_ranking'],
    margin=params['margin'],
    use_summary_image=params['use_summary_image'],
    use_se_block=params['use_se_block'],
    rollback_period=params['rollback_period'],
    rollback_lr_multiplier=params['rollback_lr_multiplier'],
    keep_prob_fn=keep_prob_fn)


def model_fn_npair(features, labels, mode, params):
  if int(params['resnet_size']) < 50:
    assert not params['use_dropblock']
    assert not params['use_resnet_d']

  dataset = data_config.get_config(params['dataset_name'])
  learning_rate_fn = learning_rate_with_decay(
    learning_rate_decay_type=params['learning_rate_decay_type'],
    batch_size=params['global_batch_size'], batch_denom=params['global_batch_size'],
    num_images=dataset.num_images['train'], num_epochs_per_decay=params['num_epochs_per_decay'],
    learning_rate_decay_factor=params['learning_rate_decay_factor'],
    end_learning_rate=params['end_learning_rate'],
    piecewise_lr_boundary_epochs=params['piecewise_lr_boundary_epochs'],
    piecewise_lr_decay_rates=params['piecewise_lr_decay_rates'],
    base_lr=params['base_learning_rate'],
    train_epochs=params['train_epochs'],
    warmup=params['lr_warmup'])

  if params['use_dropblock']:
    starter_kp = params['dropblock_kp'][0]
    end_kp = params['dropblock_kp'][1]
    batches_per_epoch = dataset.num_images['train'] / params['batch_size']
    decay_steps = int(params['train_epochs'] * batches_per_epoch)
    keep_prob_fn = keep_prob_decay(starter_kp, end_kp, decay_steps)
  else:
    keep_prob_fn = None

  steps_per_epoch = int(dataset.num_images['train'] / params['global_batch_size'])

  return run_loop_npair.resnet_model_fn(
    features=features,
    labels=labels,
    mode=mode,
    model_class=Model,
    num_classes=dataset.num_classes,
    resnet_size=params['resnet_size'],
    weight_decay=params['weight_decay'],
    learning_rate_fn=learning_rate_fn,
    momentum=params['momentum'],
    data_format=params['data_format'],
    dim_features=params['dim_features'],
    resnet_version=params['resnet_version'],
    loss_scale=params['loss_scale'],
    loss_filter_fn=None,
    dtype=params['dtype'],
    fine_tune=params['fine_tune'],
    zero_gamma=params['zero_gamma'],
    use_resnet_d=params['use_resnet_d'],
    rollback_period=params['rollback_period'],
    rollback_lr_multiplier=params['rollback_lr_multiplier'],
    use_se_block=params['use_se_block'],
    HDML_type=params['HDML_type'],
    batch_size=params['batch_size'],
    steps_per_epoch=steps_per_epoch,
    optimizer_name=params['optimizer'],
    loss_l2_reg=params['loss_l2_reg'],
    Softmax_factor=params['Softmax_factor'],
    lr_gen=params['lr_gen'],
    s_lr=params['s_lr'],
    alpha=params['alpha'],
    beta=params['beta'],
    _lambda=params['_lambda'],
    keep_prob_fn=keep_prob_fn)
