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

import inspect
import hashlib
import os
import urllib
from utils import device


################################
# log
################################
class CMDColors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_frame_info(level=2):
    caller_frame = inspect.stack()[level]
    info = inspect.getframeinfo(caller_frame[0])
    return info.filename + ':' + str(info.lineno) + ': '


class MaceLogger:
    @staticmethod
    def header(message):
        print(CMDColors.PURPLE + message + CMDColors.ENDC)

    @staticmethod
    def summary(message):
        print(CMDColors.GREEN + message + CMDColors.ENDC)

    @staticmethod
    def info(message):
        print(get_frame_info() + message)

    @staticmethod
    def warning(message):
        print(CMDColors.YELLOW + 'WARNING: ' + get_frame_info() + message
              + CMDColors.ENDC)

    @staticmethod
    def error(message):
        print(CMDColors.RED + 'ERROR: ' + get_frame_info() + message
              + CMDColors.ENDC)
        exit(1)


def mace_check(condition, message):
    if not condition:
        MaceLogger.error(message)


################################
# file
################################
def file_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_or_get_file(file,
                         sha256_checksum,
                         output_dir):
    filename = os.path.basename(file)
    output_file = "%s/%s-%s.pb" % (output_dir, filename, sha256_checksum)

    if file.startswith("http://") or file.startswith("https://"):
        if not os.path.exists(output_file) or file_checksum(
                output_file) != sha256_checksum:
            MaceLogger.info("Downloading file %s, please wait ..." % file)
            urllib.urlretrieve(file, output_file)
            MaceLogger.info("Model downloaded successfully.")
    else:
        device.execute("cp %s %s" % (file, output_file))

    return output_file
