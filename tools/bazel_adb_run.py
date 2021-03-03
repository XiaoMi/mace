# Copyright 2018 The MACE Authors. All Rights Reserved.
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
#     --target=//test/ccunit:mace_cc_test
#     --stdout_processor=stdout_processor

import argparse
import sys
import sh_commands

from common import *
from dana.dana_util import DanaUtil
from dana.dana_cli import DanaTrend
from device import DeviceWrapper, DeviceManager

TABLE_NAME = 'Ops'


def unittest_stdout_processor(stdout, device_properties, abi):
    stdout_lines = stdout.split("\n")
    for line in stdout_lines:
        if "Aborted" in line or "FAILED" in line or \
                "Segmentation fault" in line:
            raise Exception("Command failed")


def ops_benchmark_stdout_processor(stdout, dev, abi):
    stdout_lines = stdout.split("\n")
    metrics = {}
    for line in stdout_lines:
        if "Aborted" in line or "Segmentation fault" in line:
            raise Exception("Command failed")


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
        default="unittest_stdout_processor",
        help="Stdout processing function, default: stdout_processor")
    parser.add_argument(
        "--enable_neon",
        type=str2bool,
        default=True,
        help="Whether to use neon optimization")
    parser.add_argument(
        "--enable_quantize",
        type=str2bool,
        default=True,
        help="Whether to use quantization ops")
    parser.add_argument(
        "--enable_bfloat16",
        type=str2bool,
        default=True,
        help="Whether to use bfloat16")
    parser.add_argument(
        "--enable_fp16",
        type=str2bool,
        default=False,
        help="Whether to use armv8.2")
    parser.add_argument(
        "--enable_rpcmem",
        type=str2bool,
        default=False,
        help="Whether to use rpcmem")
    parser.add_argument(
        "--enable_hta",
        type=str2bool,
        default=False,
        help="Whether to use hta")
    parser.add_argument(
        '--address_sanitizer',
        action="store_true",
        help="Whether to enable AddressSanitizer")
    parser.add_argument(
        '--debug_mode',
        action="store_true",
        help="Reserve debug symbols.")
    parser.add_argument(
        "--simpleperf",
        type=str2bool,
        default=False,
        help="Whether to use simpleperf stat")
    parser.add_argument(
        '--device_yml',
        type=str,
        default='',
        help='embedded linux device config yml file')
    parser.add_argument('--vlog_level', type=int, default=0, help='vlog level')
    return parser.parse_known_args()


def report_to_dana(dana_util, item_name, metric_name,
                   device, soc, abi, value, trend):
    serie_id = dana_util.create_serie_id_lite(
        TABLE_NAME, "%s_%s_%s_%s_%s" % (metric_name, device,
                                        soc, abi, item_name))
    dana_util.report_benchmark(serie_id=serie_id, value=value, trend=trend)


def report_run_statistics(stdouts, device, soc, abi, dana_util):
    if stdouts is None:
        print('report_run_statistics failed: stdouts is None.')
        return
    for line in stdouts.split('\n'):
        line = line.strip()
        parts = line.split()
        if len(parts) == 5 and parts[0].startswith('MACE'):
            item_name = str(parts[0])
            report_to_dana(dana_util, item_name, 'time-ns', device, soc,
                           abi, float(parts[1]), DanaTrend.SMALLER)
            report_to_dana(dana_util, item_name, 'iterations', device, soc,
                           abi, float(parts[2]), DanaTrend.HIGHER)
            input_mbs = float(parts[3])
            if input_mbs < sys.float_info.min:
                input_mbs = sys.float_info.min
            report_to_dana(dana_util, item_name, 'input-MBS', device, soc,
                           abi, input_mbs, DanaTrend.HIGHER)
            gmacps = float(parts[4])
            if gmacps < sys.float_info.min:
                gmacps = sys.float_info.min
            report_to_dana(dana_util, item_name, 'GMACPS', device, soc,
                           abi, gmacps, DanaTrend.HIGHER)


def main(unused_args):
    target = FLAGS.target
    host_bin_path, bin_name = sh_commands.bazel_target_to_bin(target)
    target_abis = FLAGS.target_abis.split(',')
    dana_util = DanaUtil()

    for target_abi in target_abis:
        toolchain = infer_toolchain(target_abi)
        sh_commands.bazel_build(
            target,
            abi=target_abi,
            toolchain=toolchain,
            enable_neon=FLAGS.enable_neon,
            enable_quantize=FLAGS.enable_quantize,
            enable_bfloat16=FLAGS.enable_bfloat16,
            enable_fp16=FLAGS.enable_fp16,
            enable_rpcmem=FLAGS.enable_rpcmem,
            enable_hta=FLAGS.enable_hta,
            address_sanitizer=FLAGS.address_sanitizer,
            debug_mode=FLAGS.debug_mode)
        if FLAGS.run_target:
            target_devices = DeviceManager.list_devices(FLAGS.device_yml)
            if FLAGS.target_socs != TargetSOCTag.all and \
                    FLAGS.target_socs != TargetSOCTag.random:
                target_socs = set(FLAGS.target_socs.split(','))
                target_devices = \
                    [dev for dev in target_devices
                     if dev[YAMLKeyword.target_socs] in target_socs]
            if FLAGS.target_socs == TargetSOCTag.random:
                target_devices = sh_commands.choose_a_random_device(
                    target_devices, target_abi)

            for dev in target_devices:
                if target_abi not in dev[YAMLKeyword.target_abis]:
                    print("Skip device %s which does not support ABI %s" %
                          (dev, target_abi))
                    continue
                device_wrapper = DeviceWrapper(dev)
                stdouts = device_wrapper.run(
                    target_abi,
                    host_bin_path,
                    bin_name,
                    args=FLAGS.args,
                    opencl_profiling=True,
                    vlog_level=FLAGS.vlog_level,
                    out_of_range_check=True,
                    address_sanitizer=FLAGS.address_sanitizer,
                    simpleperf=FLAGS.simpleperf)
                globals()[FLAGS.stdout_processor](stdouts, dev, target_abi)
                if dana_util.service_available():
                    report_run_statistics(stdouts=stdouts,
                                          device=dev['device_name'],
                                          soc=dev['target_socs'],
                                          abi=target_abi, dana_util=dana_util)


if __name__ == "__main__":
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
