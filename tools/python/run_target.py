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


"""
Internal tool for mace_cc_benchmark, mace_cc_test:

python tools/python/run_target.py \
    --target_abi=armeabi-v7a --target_socs=all --target_name=mace_cc_test \
    --gtest_filter=EnvTest.*  --envs="MACE_CPP_MIN_VLOG_LEVEL=5"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from utils import device
from utils import target
from utils import config_parser
from utils import util


def run_target(target_abi, install_dir, target_obj, device_ids="all"):
    if not install_dir:
        install_dir = default_install_dir(target_abi)

    run_devices = device.choose_devices(target_abi, device_ids)

    print("Run on devices: %s" % run_devices)

    for device_id in run_devices:
        # initiate device
        dev = device.crete_device(target_abi, device_id)

        # reinstall target
        print("Install target from %s to %s" % (target_obj.path, install_dir))
        device_target = dev.install(target_obj, install_dir)
        print(device_target)

        # run on device
        print("Runing ...")
        with util.device_lock(device_id):
            dev.run(device_target)


def default_install_dir(target_abi):
    install_dir = "/tmp/mace_run"
    if target_abi == "armeabi-v7a" or target_abi == "arm64-v8a":
        install_dir = "/data/local/tmp/mace_run"

    return install_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_abi",
        type=str,
        default="armeabi-v7a",
        help="Target ABI: host, armeabi-v7a, arm64-v8a,"
             " arm-linux-gnueabihf, aarch64-linux-gnu"
    )
    parser.add_argument(
        "--target_socs",
        type=str,
        default="all",
        help="serialno for adb connection,"
             " username@ip for arm linux,"
             " host for host"
             " | all | random"
    )
    parser.add_argument(
        "--device_conf",
        type=str,
        default="",
        help="device yaml config path"
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default="mace_cc_benchmark",
        help="Target name: mace_cc_benchmark, mace_cc_test, mace_run"
    )
    parser.add_argument(
        "--build_dir",
        type=str,
        default="build/cmake-build",
        help="cmake build dir"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="if build before run"
    )

    parser.add_argument("--envs", type=str, default="",
                        help="Environment vars: "
                             " MACE_CPP_MIN_VLOG_LEVEL=2,"
                             " MACE_OUT_OF_RANGE_CHECK=1, "
                             " MACE_OPENCL_PROFILING=1,"
                             " MACE_INTERNAL_STORAGE_PATH=/path/to,"
                             " LD_PRELOAD=/path/to")

    flgs, args = parser.parse_known_args()
    return flgs, args


if __name__ == "__main__":
    flags, args = parse_args()
    if flags.device_conf:
        device_conf = config_parser.parse_device_info(flags.device_conf)
        device.ArmLinuxDevice.set_devices(device_conf)

    target_abi = flags.target_abi.strip()
    target_name = flags.target_name.strip()
    envs = flags.envs.split(" ")

    # build
    build_dir = flags.build_dir + "/" + target_abi
    if flags.build:
        cmake_shell = os.path.abspath(
            os.path.dirname(
                __file__)) + "/../cmake/cmake-build-%s.sh" % target_abi
        os.environ["BUILD_DIR"] = build_dir
        device.execute("bash " + cmake_shell)

    # run
    target = target.Target(build_dir + "/install/bin/" + target_name,
                           opts=args, envs=envs)
    run_target(target_abi, None, target, device_ids=flags.target_socs)
