from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import tensorflow as tf


def log_var_list_by_line(var_list, var_list_name):
  tf.logging.info('*********** begin %s **************', var_list_name)
  for v in var_list:
    tf.logging.info(v.name)
  tf.logging.info('*********** end   %s **************', var_list_name)


def define_log_level():
  tf_logger = logging.getLogger('tensorflow')
  tf_logger.propagate = False
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter("%(asctime)s.%(msecs).3d %(levelname).1s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
  handler.setFormatter(formatter)
  tf_logger.handlers = [handler]
  tf_logger.setLevel(tf.logging.INFO)
