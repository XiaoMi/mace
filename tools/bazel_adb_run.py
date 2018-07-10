# Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

# Must run at root dir of libmace project.
# python tools/bazel_adb_run.py \
#     --target_abis=armeabi-v7a \
#     --target_socs=sdm845
#     --target=//mace/ops:ops_test
#     --stdout_processor=stdout_processor

import argparse
import random
import re
import sys

import sh_commands


def stdout_processor(stdout, device_properties, abi):
    pass


def unittest_stdout_processor(stdout, device_properties, abi):
    stdout_lines = stdout.split("\n")
    for line in stdout_lines:
        if "Aborted" in line or "FAILED" in line or \
                "Segmentation fault" in line:
            raise Exception("Command failed")


def ops_benchmark_stdout_processor(stdout, device_properties, abi):
    stdout_lines = stdout.split("\n")
    metrics = {}
    for line in stdout_lines:
        if "Aborted" in line or "Segmentation fault" in line:
            raise Exception("Command failed")
        line = line.strip()
        parts = line.split()
        if len(parts) == 5 and parts[0].startswith("BM_"):
            metrics["%s.time_ms" % parts[0]] = str(float(parts[1]) / 1e6)
            metrics["%s.input_mb_per_sec" % parts[0]] = parts[3]
            metrics["%s.gmacc_per_sec" % parts[0]] = parts[4]

    platform = device_properties["ro.board.platform"].replace(" ", "-")
    model = device_properties["ro.product.model"].replace(" ", "-")
    tags = {
        "ro.board.platform": platform,
        "ro.product.model": model,
        "abi": abi
    }
    # sh_commands.falcon_push_metrics(server,
    #    metrics, tags=tags, endpoint="mace_ops_benchmark")


# TODO: after merge mace/python/tools and tools are merged,
# define str2bool as common util
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_abis",
        type=str,
        default="armeabi-v7a",
        help="Target ABIs, comma seperated list")
    parser.add_argument(
        "--target_socs",
        type=str,
        default="all",
        help="SoCs (ro.board.platform from getprop) to build, "
        "comma seperated list or all/random")
    parser.add_argument(
        "--target", type=str, default="//...", help="Bazel target to build")
    parser.add_argument(
        "--run_target",
        type=str2bool,
        default=False,
        help="Whether to run the target")
    parser.add_argument("--args", type=str, default="", help="Command args")
    parser.add_argument(
        "--stdout_processor",
        type=str,
        default="stdout_processor",
        help="Stdout processing function, default: stdout_processor")
    parser.add_argument(
        "--enable_neon",
        type=str2bool,
        default=True,
        help="Whether to use neon optimization")
    parser.add_argument(
        '--address_sanitizer',
        action="store_true",
        help="Whether to enable AddressSanitizer")
    return parser.parse_known_args()


def main(unused_args):
    target_socs = None
    if FLAGS.target_socs != "all" and FLAGS.target_socs != "random":
        target_socs = set(FLAGS.target_socs.split(','))
    target_devices = sh_commands.get_target_socs_serialnos(target_socs)
    if FLAGS.target_socs == "random":
        unlocked_devices = \
            [d for d in target_devices if not sh_commands.is_device_locked(d)]
        if len(unlocked_devices) > 0:
            target_devices = [random.choice(unlocked_devices)]
        else:
            target_devices = [random.choice(target_devices)]

    target = FLAGS.target
    host_bin_path, bin_name = sh_commands.bazel_target_to_bin(target)
    target_abis = FLAGS.target_abis.split(',')

    for target_abi in target_abis:
        sh_commands.bazel_build(target, abi=target_abi,
                                enable_neon=FLAGS.enable_neon,
                                address_sanitizer=FLAGS.address_sanitizer)
        if FLAGS.run_target:
            for serialno in target_devices:
                if target_abi not in set(
                        sh_commands.adb_supported_abis(serialno)):
                    print("Skip device %s which does not support ABI %s" %
                          (serialno, target_abi))
                    continue
                stdouts = sh_commands.adb_run(
                    target_abi,
                    serialno,
                    host_bin_path,
                    bin_name,
                    args=FLAGS.args,
                    opencl_profiling=True,
                    vlog_level=0,
                    device_bin_path="/data/local/tmp/mace",
                    out_of_range_check=True,
                    address_sanitizer=FLAGS.address_sanitizer)
                device_properties = sh_commands.adb_getprop_by_serialno(
                    serialno)
                globals()[FLAGS.stdout_processor](stdouts, device_properties,
                                                  target_abi)


if __name__ == "__main__":
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
