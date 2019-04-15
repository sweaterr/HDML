#!/usr/bin/env python
# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_C3_USER = "tapi"
_C3_WEB_HDFS_URL = 'http://c3.httpfs.navercorp.com:14000'
_C3_DOWNLOAD_URL = _C3_WEB_HDFS_URL + '/webhdfs/v1%s?op=open&user.name=%s'


def make_url(c3_path, c3_user=_C3_USER):
  """Get download url with c3 hdfs file path (use full path)"""
  url = _C3_DOWNLOAD_URL % (c3_path, c3_user)
  return url
