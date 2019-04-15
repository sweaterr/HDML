# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from official.utils.misc import distribution_utils

def _monkey_patch_org_assert_broadcastable():
  """Monkey-patch `assert_broadcast` op to avoid OOM when enabling XLA."""
  def no_op_assert_broadcastable(weights, values):
    del weights, values
    tf.logging.info(
        'Using monkey-patched version of assert_broadcastable op, which always '
        'returns an no_op. It should be removed after XLA OOM issue is fixed.')
    return tf.constant([], dtype=tf.float32)

  from tensorflow.python.ops import weights_broadcast_ops  # pylint: disable=g-import-not-at-top
  if not hasattr(weights_broadcast_ops, 'org_assert_broadcastable'):
    weights_broadcast_ops.org_assert_broadcastable = (
        weights_broadcast_ops.assert_broadcastable)
  weights_broadcast_ops.assert_broadcastable = no_op_assert_broadcastable


def get_session_config(flags_obj):
  """Return config proto according to flag settings, or None to use default."""
  config = tf.ConfigProto(
    inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
    intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
    allow_soft_placement=True)

  if flags_obj.xla_type == 1:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  elif flags_obj.xla_type == 2:
    # TODO(haoyuzhang): Remove this monkey patch when XLA OOM issue is fixed.
    _monkey_patch_org_assert_broadcastable()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
    # Disable PinToHostOptimizer in grappler when enabling XLA because it causes
    # OOM and performance regression.
    config.graph_options.rewrite_options.pin_to_host_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)
  return config

def get_run_config(flags_obj, flags_core, session_config, num_images_train):
  distribution_strategy = distribution_utils.get_distribution_strategy(
    flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

  steps_per_epoch = flags_obj.save_checkpoints_epochs \
                    * int(num_images_train // int(flags_obj.batch_size))

  run_config = tf.estimator.RunConfig(
    train_distribute=distribution_strategy, session_config=session_config,
    keep_checkpoint_max=flags_obj.keep_checkpoint_max,
    save_checkpoints_steps=int(steps_per_epoch),
    save_checkpoints_secs=None,
  )
  return run_config