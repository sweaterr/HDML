# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

try:
  from functions import model_fns
except:
  from functions import *
from utils import log_utils
from utils import rollback_utils


class WarmStartHook(tf.train.SessionRunHook):
  def __init__(self, checkpoint_path):
    self.checkpoint_path = checkpoint_path
    self.initialized = False
    self.var_list_warm_start = []
    self.saver = None

  def begin(self):
    var_list_all = tf.trainable_variables()

    for v in var_list_all:
      if 'dense' in v.name and not 'se_block' in v.name:
        continue
      elif 'Classifier' in v.name:
        continue
      elif 'Generator' in v.name:
        continue
      elif 'Softmax_classifier' in v.name:
        continue
      else:
        self.var_list_warm_start.append(v)

    if tf.gfile.IsDirectory(self.checkpoint_path):
      self.checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)

    if not self.initialized:
      self.saver = tf.train.Saver(self.var_list_warm_start)
      self.initialized = True

  def after_create_session(self, session, coord=None):
    tf.logging.info('Session created.')
    if self.checkpoint_path and session.run(tf.train.get_or_create_global_step()) == 0:
      log_utils.log_var_list_by_line(self.var_list_warm_start, 'var_list_warm_start')
      tf.logging.info('Fine-tuning from %s' % self.checkpoint_path)
      self.saver.restore(session, self.checkpoint_path)



class RollbackHook(tf.train.SessionRunHook):
  """
  각 학습 period에 맞게 layer를 pretrained 변수로 다시 Rolling Back 한다.
  """

  def __init__(self,
               pretrained_checkpoint_path,
               rollback_keep_checkpoint_path,
               rollback_period,
               resnet_size,
               layer_sindex,
               bottleneck,
               resnet_version):
    self.pretrained_checkpoint_path = pretrained_checkpoint_path
    self.rollback_keep_checkpoint_path = rollback_keep_checkpoint_path
    self.rollback_period = rollback_period
    self.initialized = False
    self.resnet_size = resnet_size
    self.layer_sindex = layer_sindex
    self.bottleneck = bottleneck
    self.resnet_version = resnet_version
    self.var_list_rollback = []
    self.var_list_keep = []
    self.saver_rollback = None
    self.saver_keep = None

  def begin(self):
    # if not self.initialized:
    var_list_all = tf.contrib.framework.get_trainable_variables()
    if self.rollback_period == 1:  # 처음에는 모든 param을 warm-start
      for v in var_list_all:
        if 'dense' in v.name and not 'se_block' in v.name:
          continue
        else:
          self.var_list_rollback.append(v)
    else:
      if self.rollback_period == 2:
        restore_blocks_list = [2, 3, 4]
      elif self.rollback_period == 3:
        restore_blocks_list = [3, 4]
      elif self.rollback_period == 4:
        restore_blocks_list = [4]
      else:
        raise ValueError("period must be a integer in the range [0, 3], got %d" % self.period)

      block_sizes = model_fns.get_block_sizes(self.resnet_size)
      targets = rollback_utils.get_restore_targets_list(block_sizes,
                                                        restore_blocks_list,
                                                        self.layer_sindex,
                                                        self.bottleneck,
                                                        self.resnet_version)
      for v in var_list_all:
        if any(x in v.name for x in targets):
          self.var_list_rollback.append(v)
        else:
          self.var_list_keep.append(v)

    if tf.gfile.IsDirectory(self.pretrained_checkpoint_path):
      self.pretrained_checkpoint_path = tf.train.latest_checkpoint(self.pretrained_checkpoint_path)
    else:
      self.pretrained_checkpoint_path = self.pretrained_checkpoint_path

    if tf.gfile.IsDirectory(self.rollback_keep_checkpoint_path):
      self.rollback_keep_checkpoint_path = tf.train.latest_checkpoint(self.rollback_keep_checkpoint_path)
    else:
      self.rollback_keep_checkpoint_path = self.rollback_keep_checkpoint_path

    if len(self.var_list_rollback) != 0:
      self.saver_rollback = tf.train.Saver(self.var_list_rollback)
    if len(self.var_list_keep) != 0:
      self.saver_keep = tf.train.Saver(self.var_list_keep)

  def after_create_session(self, session, coord=None):
    if session.run(tf.train.get_or_create_global_step()) == 0:
      tf.logging.info('Session created.')
      log_utils.log_var_list_by_line(self.var_list_rollback, 'var_list_rollback')
      log_utils.log_var_list_by_line(self.var_list_keep, 'var_list_keep')
      if self.saver_rollback:
        self.saver_rollback.restore(session, self.pretrained_checkpoint_path)
      if self.saver_keep:
        self.saver_keep.restore(session, self.rollback_keep_checkpoint_path)
