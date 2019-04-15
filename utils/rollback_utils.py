# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.warm_start_utils import *


def get_keep_target_name(block_sizes, block, layer_sindex, bottleneck, resnet_version):
  block_sindex = 0
  block_eindex, _ = get_conv2d_layer_ranges(block_sizes, block, layer_sindex, bottleneck)

  block_bn_sindex = 0
  block_bn_eindex, _ = get_batch_norm_layer_ranges(block_sizes, block, layer_sindex, bottleneck,
                                                   resnet_version=resnet_version)

  block_se_sindex = 0
  block_se_eindex = get_se_block_ranges(block_sizes, block)

  targets = []
  for i in range(block_sindex, block_eindex):
    if i == 0:
      targets.append('conv2d/')
    else:
      targets.append('conv2d_%d/' % i)

  for i in range(block_bn_sindex, block_bn_eindex):
    if i == 0:
      targets.append('batch_normalization/')
    else:
      targets.append('batch_normalization_%d/' % i)

  for i in range(block_se_sindex, block_se_eindex):
    if i == 0:
      targets.append('se_block/')
    else:
      targets.append('se_block_%d/' % i)

  return targets


def get_block_target_name(block_sizes, block_num, layer_sindex, bottleneck, resnet_version):
  block1_sindex, _ = get_conv2d_layer_ranges(block_sizes, block_num, layer_sindex, bottleneck)
  if block_num == 4:
    _, block1_eindex = get_conv2d_layer_ranges(block_sizes, block_num, layer_sindex, bottleneck)
    block1_eindex += 1  # 이게 없으면 밑에서 ranges 시 끝에 하나를 덜 돈다.
  else:
    block1_eindex, _ = get_conv2d_layer_ranges(block_sizes, block_num + 1, layer_sindex, bottleneck)

  block1_bn_sindex, _ = get_batch_norm_layer_ranges(block_sizes, block_num, layer_sindex, bottleneck,
                                                    resnet_version=resnet_version)
  if block_num == 4:
    _, block1_bn_eindex = get_batch_norm_layer_ranges(block_sizes, block_num, layer_sindex, bottleneck,
                                                      resnet_version=resnet_version)
    block1_bn_eindex += 1  # 이게 없으면 밑에서 ranges 시 끝에 하나를 덜 돈다.
  else:
    block1_bn_eindex, _ = get_batch_norm_layer_ranges(block_sizes, block_num + 1, layer_sindex, bottleneck,
                                                      resnet_version=resnet_version)

  block1_se_sindex = get_se_block_ranges(block_sizes, block_num)
  block1_se_eindex = get_se_block_ranges(block_sizes, block_num + 1)

  targets = []
  for i in range(block1_sindex, block1_eindex):
    if i == 0:
      targets.append('conv2d/')
    else:
      targets.append('conv2d_%d/' % i)

  for i in range(block1_bn_sindex, block1_bn_eindex):
    if i == 0:
      targets.append('batch_normalization/')
    else:
      targets.append('batch_normalization_%d/' % i)

  for i in range(block1_se_sindex, block1_se_eindex):
    if i == 0:
      targets.append('se_block/')
    else:
      targets.append('se_block_%d/' % i)

  return targets


def get_restore_targets_list(block_sizes, blocks, layer_sindex, bottleneck, resnet_version):
  """
  Example:
    get_restore_var_list([3, 4, 6, 3], [3,4], 3, True, 1)
  """
  targets = []
  for block in blocks:
    targets.extend(get_block_target_name(block_sizes, block, layer_sindex, bottleneck, resnet_version))
  return targets


def get_rollback_varlist(var_list, period,
                         block_sizes,
                         layer_sindex,
                         bottleneck=True,
                         resnet_version=1):
  """
  period를 전달하면, x0.1, x1.0, x10 variable을 리턴한다
  period
  """
  assert 0 < period < 5
  lr_fc_var = []
  lr_rollback_var = []
  lr_keep_var = []
  if period == 1:
    # B1,2,3,4 0.01 => 0.001
    # FC 0.1 => 0.01
    for v in var_list:
      if 'dense' in v.name and not 'se_block' in v.name:
        lr_fc_var.append(v)
      else:
        lr_rollback_var.append(v)
  else:
    # period 2
    # B1,FC 0.001 => 0.0001
    # B2,3,4 0.01 => 0.001
    # period 3
    # B1,2,FC 0.001 => 0.0001
    # B3,4 0.01 => 0.001
    # period 4
    # B1,2,3 FC 0.001 => 0.0001
    # B4 0.01 => 0.001
    targets = get_keep_target_name(block_sizes, period, layer_sindex, bottleneck, resnet_version)

    tf.logging.debug("keep_target_name")
    for t in targets:
      tf.logging.debug(t)

    for v in var_list:
      if 'dense' in v.name and not 'se_block' in v.name:
        lr_keep_var.append(v)
      elif any(x in v.name for x in targets):
        lr_keep_var.append(v)
      else:
        lr_rollback_var.append(v)

  return lr_fc_var, lr_rollback_var, lr_keep_var
