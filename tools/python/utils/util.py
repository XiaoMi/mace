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
import filelock
import errno
import os
import sys
import shutil
import traceback


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
        print(CMDColors.PURPLE + str(message) + CMDColors.ENDC)

    @staticmethod
    def summary(message):
        print(CMDColors.GREEN + str(message) + CMDColors.ENDC)

    @staticmethod
    def info(message):
        print(get_frame_info() + str(message))

    @staticmethod
    def warning(message):
        print(CMDColors.YELLOW + 'WARNING: ' + get_frame_info() + str(message)
              + CMDColors.ENDC)

    @staticmethod
    def error(message, level=2):
        print(CMDColors.RED + 'ERROR: ' + get_frame_info(level) + str(message)
              + CMDColors.ENDC)
        exit(1)


def mace_check(condition, message):
    if not condition:
        for line in traceback.format_stack():
            print(line.strip())

        MaceLogger.error(message, level=3)


################################
# String Formatter
################################
class StringFormatter:
    @staticmethod
    def table(header, data, title, align="R"):
        data_size = len(data)
        column_size = len(header)
        column_length = [len(str(ele)) + 1 for ele in header]
        for row_idx in range(data_size):
            data_tuple = data[row_idx]
            ele_size = len(data_tuple)
            assert (ele_size == column_size)
            for i in range(ele_size):
                column_length[i] = max(column_length[i],
                                       len(str(data_tuple[i])) + 1)

        table_column_length = sum(column_length) + column_size + 1
        dash_line = '-' * table_column_length + '\n'
        header_line = '=' * table_column_length + '\n'
        output = ""
        output += dash_line
        output += str(title).center(table_column_length) + '\n'
        output += dash_line
        output += '|' + '|'.join([str(header[i]).center(column_length[i])
                                  for i in range(column_size)]) + '|\n'
        output += header_line

        for data_tuple in data:
            ele_size = len(data_tuple)
            row_list = []
            for i in range(ele_size):
                if align == "R":
                    row_list.append(str(data_tuple[i]).rjust(column_length[i]))
                elif align == "L":
                    row_list.append(str(data_tuple[i]).ljust(column_length[i]))
                elif align == "C":
                    row_list.append(str(data_tuple[i])
                                    .center(column_length[i]))
            output += '|' + '|'.join(row_list) + "|\n" + dash_line
        return output

    @staticmethod
    def block(message):
        line_length = 10 + len(str(message)) + 10
        star_line = '*' * line_length + '\n'
        return star_line + str(message).center(line_length) + '\n' + star_line


def formatted_file_name(input_file_name, input_name):
    res = input_file_name + '_'
    for c in input_name:
        res += c if c.isalnum() else '_'
    return res


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
                         output_file):
    if file.startswith("http://") or file.startswith("https://"):
        if not os.path.exists(output_file) or file_checksum(
                output_file) != sha256_checksum:
            MaceLogger.info("Downloading file %s to %s, please wait ..."
                            % (file, output_file))
            if sys.version_info >= (3, 0):
                import urllib.request
                data = urllib.request.urlopen(file)
                out_handle = open(output_file, "wb")
                out_handle.write(data.read())
                out_handle.close()
            else:
                import urllib
                urllib.urlretrieve(file, output_file)
            MaceLogger.info("Model downloaded successfully.")
    else:
        shutil.copyfile(file, output_file)

    if sha256_checksum:
        mace_check(file_checksum(output_file) == sha256_checksum,
                   "checksum validate failed")

    return output_file


def download_or_get_model(file,
                          sha256_checksum,
                          output_dir):
    filename = os.path.basename(file)
    output_file = "%s/%s-%s.pb" % (output_dir, filename, sha256_checksum)
    download_or_get_file(file, sha256_checksum, output_file)
    return output_file


################################
# bazel commands
################################
class ABIType(object):
    armeabi_v7a = 'armeabi-v7a'
    arm64_v8a = 'arm64-v8a'
    arm64 = 'arm64'
    aarch64 = 'aarch64'
    armhf = 'armhf'
    host = 'host'


def abi_to_internal(abi):
    if abi in [ABIType.armeabi_v7a, ABIType.arm64_v8a]:
        return abi
    if abi == ABIType.arm64:
        return ABIType.aarch64
    if abi == ABIType.armhf:
        return ABIType.armeabi_v7a


################################
# lock
################################
def device_lock(device_id, timeout=7200):
    return filelock.FileLock("/tmp/device-lock-%s" % device_id,
                             timeout=timeout)


def is_device_locked(device_id):
    try:
        with device_lock(device_id, timeout=0.000001):
            return False
    except filelock.Timeout:
        return True


################################
# os
################################
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
