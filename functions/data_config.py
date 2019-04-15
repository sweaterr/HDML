from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Default(object):
  shuffle_buffer = 1000
  default_image_size = 224
  num_channels = 3
  num_train_files = 128
  num_val_files = 16


class Food100(Default):
  num_classes = 100
  num_images = {
    'train': 11525,
    'validation': 2836,
  }
  dataset_name = 'food100'


class Food101(Default):
  num_classes = 101
  num_images = {
    'train': 75750,
    'validation': 25250,
  }
  dataset_name = 'food101'


class NAVERFood321(Default):
  num_classes = 321
  num_images = {
    'train': 264074,
    'validation': 5733,
  }
  dataset_name = 'naver_food321'


class NAVERFood547(Default):
  num_classes = 547
  num_images = {
    'train': 194761,
    'validation': 8943,
  }
  dataset_name = 'naver_food547'


class NAVERFood581(Default):
  num_classes = 581
  num_images = {
    'train': 184094,
    'validation': 45849,
  }
  dataset_name = 'naver_food581'


class ImageNet(Default):
  shuffle_buffer = 10000
  num_classes = 1001
  num_images = {
    'train': 1281167,
    'validation': 50000,
  }
  num_train_files = 1024
  dataset_name = 'imagenet'

class CUB_200_2011(Default):
  shuffle_buffer = 100
  num_classes = 100
  num_images = {
    'train': 5864,
    'validation': 5924,
  }
  num_train_files = 100
  dataset_name = 'cub_200_2011'

class CARS196(Default):
  shuffle_buffer = 100
  num_classes = 98 # zero-shot setting이라서, 전체 클래스의 반이다.
  num_images = {
    'train': 8054,
    'validation': 8131,
  }
  dataset_name = 'cars196'

class NAVERPlant798(Default):
  shuffle_buffer = 10000
  num_classes = 798
  num_images = {
    'train': 141709,
    'validation': 1583,
  }
  num_train_files = 128
  num_val_files = 16
  dataset_name = 'naver_plant798'

def get_config(data_name):
  if data_name == 'food100':
    return Food100()
  elif data_name == 'food101':
    return Food101()
  elif data_name == 'imagenet':
    return ImageNet()
  elif data_name == 'naver_food321':
    return NAVERFood321()
  elif data_name == 'naver_food547':
    return NAVERFood547()
  elif data_name == 'naver_food581':
    return NAVERFood581()
  elif data_name == 'cub_200_2011':
    return CUB_200_2011()
  elif data_name == 'cars196':
    return CARS196()
  elif data_name == 'naver_plant798':
    return NAVERPlant798()
  else:
    raise ValueError("Unable to support {} dataset.".format(data_name))
