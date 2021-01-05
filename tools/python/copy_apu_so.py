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

from utils import device
from utils.util import MaceLogger
from utils.util import mace_check


def get_apu_so_paths_by_props(android_ver, target_soc):
    so_path_array = []
    so_path = "third_party/apu/"
    so_path += "android_Q/" if android_ver <= 10 else "android_R/"
    if android_ver <= 10:
        if target_soc.startswith("mt67"):
            so_path += "mt67xx/"
        elif target_soc.startswith("mt68"):
            so_path += "mt68xx/"
        else:
            mace_check(False, "the soc is not supported: %s" % target_soc)
        frontend_so_path = so_path + "%s/libapu-frontend.so" % target_soc
        if not os.path.exists(frontend_so_path):
            frontend_so_path = so_path + "libapu-frontend.so"
        so_path_array.append(frontend_so_path)
        so_path_array.append(so_path + "%s/libapu-platform.so" % target_soc)
    else:
        so_path_array.append(so_path + "libapu-apuwareapusys.mtk.so")
        so_path_array.append(so_path + "libapu-apuwareutils.mtk.so")
        so_path_array.append(so_path + "libapu-apuwarexrp.mtk.so")
        so_path_array.append(so_path + "libapu-frontend.so")
        so_path_array.append(so_path + "libapu-platform.so")
    return so_path_array


def get_apu_so_paths(android_device):
    target_props = android_device.info()
    target_soc = target_props["ro.board.platform"]
    android_ver = (int)(target_props["ro.build.version.release"])
    return get_apu_so_paths_by_props(android_ver, target_soc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_abi",
        type=str,
        default="arm64-v8a",
        help="Target ABI: only support arm64-v8a"
    )
    parser.add_argument(
        "--target_soc",
        type=str,
        default="all",
        help="serialno for adb connection"
    )
    parser.add_argument(
        "--apu_path",
        type=str,
        default="",
        help="path for storing apu so files on device"
    )

    return parser.parse_known_args()


if __name__ == "__main__":
    flags, args = parse_args()
    run_devices = device.choose_devices(flags.target_abi, flags.target_soc)
    device_num = len(run_devices)
    if device_num == 0:  # for CI
        MaceLogger.warning("No Android devices are plugged in, "
                           "you need to copy `apu` so files by yourself.")
    elif device_num > 1:  # for CI
        MaceLogger.warning("More than one Android devices are plugged in, "
                           "you need to copy `apu` so files by yourself.")
    else:
        device_id = run_devices[0]
        android_device = device.create_device(flags.target_abi, device_id)
        apu_so_paths = get_apu_so_paths(android_device)
        for apu_so_path in apu_so_paths:
            device.execute("cp -f %s %s" % (apu_so_path, flags.apu_path), True)
