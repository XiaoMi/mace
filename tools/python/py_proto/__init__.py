# Copyright 2019 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from utils import device
from utils.util import MaceLogger

cwd = os.path.dirname(__file__)

# TODO: Remove bazel deps
try:
    device.execute("bazel build //mace/proto:mace_py")
    device.execute("cp -f bazel-genfiles/mace/proto/mace_pb2.py %s" % cwd)

    device.execute("bazel build //third_party/caffe:caffe_py")
    device.execute(
        "cp -f bazel-genfiles/third_party/caffe/caffe_pb2.py %s" % cwd)
except:  # noqa
    MaceLogger.warning("No bazel, use cmake.")
