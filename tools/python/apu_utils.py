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

import argparse
import os
import sys

from utils import device
from utils.util import MaceLogger
from utils.util import mace_check


def get_apu_version(enable_apu, android_ver, target_soc):
    if enable_apu:
        android_ver = (int)(android_ver)
        if android_ver <= 10:  # android Q
            target_soc = target_soc.lower()
            if target_soc.startswith("mt67"):
                return 1
            else:
                return 2
        elif android_ver == 11:  # android R
            target_soc = target_soc.lower()
            if target_soc.startswith("mt689") or target_soc == "mt6877":
                return 4
            else:
                return 3
        else:  # android S
            return 4
    return -1


def get_apu_so_paths_by_props(android_ver, target_soc):
    so_path_array = []
    apu_version = get_apu_version(True, android_ver, target_soc)
    so_path = "third_party/apu/"
    if apu_version == 1 or apu_version == 2:
        if apu_version == 1:
            so_path += "android_Q/mt67xx/"
        else:
            so_path += "android_Q/mt68xx/"
        frontend_so_path = so_path + "%s/libapu-frontend.so" % target_soc
        if not os.path.exists(frontend_so_path):
            frontend_so_path = so_path + "libapu-frontend.so"
        so_path_array.append(frontend_so_path)
        so_path_array.append(so_path + "%s/libapu-platform.so" % target_soc)
    elif apu_version == 3:
        so_path += "android_R/"
        # For android R except mt689x&mt6877
        if target_soc != "mt6785":
            so_path_array.append(so_path + "libapu-apuwareapusys.mtk.so")
        so_path_array.append(so_path + "libapu-apuwareutils.mtk.so")
        so_path_array.append(so_path + "libapu-apuwarexrp.mtk.so")
        so_path_array.append(so_path + "libapu-frontend.so")
        so_path_array.append(so_path + "libapu-platform.so")
    else:  # For android S and mt689x&mt6877 on android R
        mace_check(apu_version == 4, "Invalid apu verison")

    return so_path_array


def get_apu_so_paths(android_device):
    target_props = android_device.info()
    target_soc = target_props["ro.board.platform"]
    android_ver = (int)(target_props["ro.build.version.release"])
    return get_apu_so_paths_by_props(android_ver, target_soc)


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--target_abi",
        type=str,
        default="arm64-v8a",
        help="Target ABI: only support arm64-v8a"
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    version_parser = subparsers.add_parser(
        'get-version',
        parents=[base_parser],
        help='get apu version')
    version_parser.set_defaults(func=get_version)

    copy_so_parser = subparsers.add_parser(
        'copy-so-files',
        parents=[base_parser],
        help='copy apu files to apu_path')
    copy_so_parser.add_argument(
        "--apu_path",
        type=str,
        default="",
        help="path for storing apu so files on device"
    )
    copy_so_parser.set_defaults(func=copy_so_files)

    return parser.parse_known_args()


def get_cur_device_id(flags):
    run_devices = device.choose_devices(flags.target_abi, "all")
    run_device = None
    device_num = len(run_devices)
    if device_num == 0:  # for CI
        MaceLogger.warning("No Android devices are plugged in, "
                           "you need to copy `apu` so files by yourself.")
    elif device_num > 1:  # for CI
        MaceLogger.warning("More than one Android devices are plugged in, "
                           "you need to copy `apu` so files by yourself.")
    else:
        run_device = run_devices[0]

    return run_device


def get_version(flags):
    device_id = get_cur_device_id(flags)
    if device_id is not None:
        android_device = device.create_device(flags.target_abi, device_id)
        target_props = android_device.info()
        target_soc = target_props["ro.board.platform"]
        android_ver = (int)(target_props["ro.build.version.release"])
        apu_version = get_apu_version(True, android_ver, target_soc)
    else:
        apu_version = 4
        MaceLogger.warning("Can not get unique device ID, MACE select the"
                           " latest apu version: %s" % apu_version)
    sys.exit(apu_version)


def copy_so_files(flags):
    apu_so_paths = []
    device_id = get_cur_device_id(flags)
    if device_id is not None:
        android_device = device.create_device(flags.target_abi, device_id)
        apu_so_paths = get_apu_so_paths(android_device)
    for apu_so_path in apu_so_paths:
        device.execute("cp -f %s %s" % (apu_so_path, flags.apu_path), True)


if __name__ == "__main__":
    flags, args = parse_args()
    flags.func(flags)
