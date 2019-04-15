# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_MAX_BLOCK_NUM = 4


def get_conv2d_layer_ranges(block_sizes, block_layer_index, conv_layer_sindex, bottleneck):
  """Retrieve the conv2d layer range of the specific block_layer point.

  Args:
    block_sizes: The size of each block_layer in the Resnet model.
    block_layer_index: The specific index as they start with.
    conv_layer_sindex: The specific index as conv layers start with.
    bottleneck: If True, use "bottleneck" layers

  Returns:
    A pair of conv2d layer ranges.
  """
  num_conv2d = 3 if bottleneck else 2

  def _helper(ret, start_index, xs):
    if not xs:
      return ret
    else:
      head, tail = xs[0], xs[1:]
      end_index = start_index + (1 + head * num_conv2d) - 1
      ret.append((start_index, end_index))

      return _helper(ret, end_index + 1, tail)

  block_ranges = _helper([], conv_layer_sindex, block_sizes)
  grad_block_ranges = block_ranges[(block_layer_index - 1):]

  start_index = min(grad_block_ranges, key=lambda t: t[0])[0]
  end_index = max(grad_block_ranges, key=lambda t: t[1])[1]

  return (start_index, end_index)


def get_batch_norm_layer_ranges(block_sizes, block_layer_index, bn_layer_sindex,
                                bottleneck, resnet_version):
  """Retrieve the batch normalization layer range of the specific block_layer point.

  Args:
    block_sizes: The size of each block_layer in the Resnet model.
    block_layer_index: The specific index as they start with.
    bn_layer_sindex: The specific index as bn layers start with.
    bottleneck: If True, use "bottleneck" layers
    resnet_version: Integer representing which version of the ResNet network
      to use. Valid values: [1, 2]

  Returns:
    A pair of batch normalization layer ranges.
  """
  if resnet_version == 1:
    return get_conv2d_layer_ranges(block_sizes, block_layer_index, bn_layer_sindex, bottleneck)
  else:
    bottleneck_mul = 3 if bottleneck else 2

    def _helper(ret, start_index, xs):
      if not xs:
        return ret
      else:
        head, tail = xs[0], xs[1:]
        end_index = start_index + (head * bottleneck_mul) - 1
        ret.append((start_index, end_index))

        return _helper(ret, end_index + 1, tail)

    block_ranges = _helper([], bn_layer_sindex, block_sizes)
    grad_block_ranges = block_ranges[(block_layer_index - 1):]

    start_index = min(grad_block_ranges, key=lambda t: t[0])[0]
    end_index = max(grad_block_ranges, key=lambda t: t[1])[1]

    return (start_index, end_index)


def get_se_block_ranges(block_sizes, block_layer_index):
  idx = 0
  for i in range(block_layer_index - 1):
    idx += block_sizes[i]
  sindex = idx
  return sindex


def block_layer_grad_filter(gvs, block_sizes, block_layer_index,
                            bottleneck, resnet_version=1, use_resnet_d=False):
  """Apply gradient updates to the specific block layers.

  This function is used for fine tuning.

  Args:
    gvs: list of tuples with gradients and variable info.
    block_sizes: A list contaning n values, where n is the number of sets of
      block layers desired. Each value should be the number of blocks in the
      i-th set.
    block_layer_index: Block layer starting point for applying gradients. If greater than block_sizes's length, only return the dense layer's gradients.
    bottleneck: If True, use "bottleneck" layers.
    resnet_version: resnet version.
    use_resnet_d: If True, use resnet_d architecture.

  Returns:
    filtered gradients so that only the specific layer remains.
  """

  if use_resnet_d and resnet_version == 1:
    conv_layer_sindex = 3
    bn_layer_sindex = 3
  else:
    conv_layer_sindex = 1
    bn_layer_sindex = 1

  targets = ['dense/']
  if len(block_sizes) >= block_layer_index:
    conv2d_sindex, conv2d_eindex = get_conv2d_layer_ranges(block_sizes, block_layer_index,
                                                           conv_layer_sindex, bottleneck)
    tf.logging.debug("conv2d layer range: ({}, {})".format(conv2d_sindex, conv2d_eindex))

    for i in range(conv2d_sindex, conv2d_eindex + 1):
      targets.append('conv2d_%d/' % i)

    batch_norm_sindex, batch_norm_eindex = get_batch_norm_layer_ranges(block_sizes,
                                                                       block_layer_index,
                                                                       bn_layer_sindex,
                                                                       bottleneck,
                                                                       resnet_version)
    tf.logging.debug("batch norm op range: ({}, {})".format(batch_norm_sindex,
                                                            batch_norm_eindex))

    for i in range(batch_norm_sindex, batch_norm_eindex + 1):
      targets.append('batch_normalization_%d/' % i)

    block1_se_sindex = get_se_block_ranges(block_sizes, block_layer_index)
    block1_se_eindex = get_se_block_ranges(block_sizes, _MAX_BLOCK_NUM + 1)

    for i in range(block1_se_sindex, block1_se_eindex):
      if i == 0:
        targets.append('se_block/')
      else:
        targets.append('se_block_%d/' % i)

  if isinstance(gvs[0], tuple):
    return [(g, v) for g, v in gvs if any(x in v.name for x in targets)]
  else:
    return [v for v in gvs if any(x in v.name for x in targets)]
