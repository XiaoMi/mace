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

import numpy as np
import sys
import subprocess
import time
import sh
import yaml

import common
from common import *
from dana.dana_cli import DanaTrend
from dana.dana_util import DanaUtil
from six.moves import urllib

sys.path.insert(0, "tools/python")  # noqa
from utils import config_parser
import layers_validate
import sh_commands

sys.path.insert(0, "tools/python")  # noqa
import apu_utils
from hexagon_soc_defines import get_soc_skel_info
from hexagon_soc_defines import DspType


class DeviceWrapper:
    allow_scheme = ('ssh', 'adb')

    def __init__(self, device_dict):
        """
        init device with device dict info
        :type device_dict: Device
        :param device_dict安卓跑通mace实例: a key-value dict that holds the device information,
                       which attribute has:
                       device_name, target_abis, target_socs, system,
                        address, username
        """
        self._dana_util = DanaUtil()
        diff = set(device_dict.keys()) - set(YAMLKeyword.__dict__.keys())
        if len(diff) > 0:
            six.print_('Wrong key detected: ')
            six.print_(diff)
            raise KeyError(str(diff))
        self.__dict__.update(device_dict)
        if self.system == SystemType.android:
            self.data_dir = PHONE_DATA_DIR
            self.interior_dir = self.data_dir + '/interior'
        elif self.system == SystemType.arm_linux:
            try:
                sh.ssh('-q', '{}@{}'.format(self.username, self.address),
                       'exit')
            except sh.ErrorReturnCode as e:
                six.print_('device connect failed, '
                           'please check your authentication')
                raise e
            self.data_dir = DEVICE_DATA_DIR
            self.interior_dir = self.data_dir + '/interior'
        elif self.system == SystemType.host:
            self.data_dir = DEVICE_DATA_DIR
            self.interior_dir = self.data_dir + '/interior'

    ##################
    #  internal use  #
    ##################

    def exec_command(self, command, *args, **kwargs):
        if self.system == SystemType.android:
            sh.adb('-s', self.address, 'shell', command, *args, **kwargs)
        elif self.system == SystemType.arm_linux:
            sh.ssh('{}@{}'.format(self.username, self.address),
                   command, *args, **kwargs)

    #####################
    #  public interface #
    #####################

    def is_lock(self):
        return sh_commands.is_device_locked(self.address)

    def lock(self):
        return sh_commands.device_lock(self.address)

    def clear_data_dir(self):
        if self.system == SystemType.android:
            sh_commands.clear_phone_data_dir(self.address, PHONE_DATA_DIR)
        elif self.system == SystemType.arm_linux:
            self.exec_command('rm -rf {}'.format(self.data_dir))

    def pull_from_data_dir(self, filename, dst_path):
        if self.system == SystemType.android:
            self.pull(PHONE_DATA_DIR, filename, dst_path)
        elif self.system == SystemType.arm_linux:
            self.pull(DEVICE_DATA_DIR, filename, dst_path)

    def create_internal_storage_dir(self):
        internal_storage_dir = '{}/interior/'.format(self.data_dir)
        if self.system == SystemType.android:
            sh_commands.create_internal_storage_dir(self.address,
                                                    internal_storage_dir)
        elif self.system == SystemType.arm_linux:
            self.exec_command('mkdir -p {}'.format(internal_storage_dir))
        return internal_storage_dir

    def rm(self, file):
        if self.system == SystemType.android:
            sh.adb('-s', self.address, 'shell', 'rm', '-rf', file, _fg=True)
        elif self.system == SystemType.arm_linux:
            self.exec_command('rm -rf {}'.format(file), _fg=True)

    def push(self, src_path, dst_path):
        mace_check(os.path.exists(src_path), "Device",
                   '{} not found'.format(src_path))
        six.print_("Push %s to %s" % (src_path, dst_path))
        if self.system == SystemType.android:
            sh_commands.adb_push(src_path, dst_path, self.address)
        elif self.system == SystemType.arm_linux:
            try:
                sh.scp(src_path, '{}@{}:{}'.format(self.username,
                                                   self.address,
                                                   dst_path))
            except sh.ErrorReturnCode_1 as e:
                six.print_('Push Failed !', e, file=sys.stderr)
                raise e

    def get_apu_version(self, enable_apu):
        if enable_apu:
            target_props = sh_commands.adb_getprop_by_serialno(self.address)
            target_soc = target_props["ro.board.platform"]
            android_ver = (int)(target_props["ro.build.version.release"])
            return apu_utils.get_apu_version(enable_apu, android_ver,
                                             target_soc)
        else:
            return -1

    def get_apu_so_paths(self):
        target_props = sh_commands.adb_getprop_by_serialno(self.address)
        target_soc = target_props["ro.board.platform"]
        android_ver = (int)(target_props["ro.build.version.release"])
        return apu_utils.get_apu_so_paths_by_props(android_ver, target_soc)

    def pull(self, src_path, file_name, dst_path='.'):
        if not os.path.exists(dst_path):
            sh.mkdir("-p", dst_path)
        src_file = "%s/%s" % (src_path, file_name)
        dst_file = "%s/%s" % (dst_path, file_name)
        if os.path.exists(dst_file):
            sh.rm('-f', dst_file)
        six.print_("Pull %s to %s" % (src_file, dst_path))
        if self.system == SystemType.android:
            sh_commands.adb_pull(
                src_file, dst_file, self.address)
        elif self.system == SystemType.arm_linux:
            try:
                sh.scp('-r', '%s@%s:%s' % (self.username,
                                           self.address,
                                           src_file),
                       dst_file)
            except sh.ErrorReturnCode_1 as e:
                six.print_("Pull Failed !", file=sys.stderr)
                raise e

    def tuning_run(self,
                   abi,
                   target_dir,
                   target_name,
                   vlog_level,
                   embed_model_data,
                   model_output_dir,
                   input_nodes,
                   output_nodes,
                   input_shapes,
                   input_data_types,
                   output_data_types,
                   input_data_formats,
                   output_shapes,
                   output_data_formats,
                   mace_model_dir,
                   model_tag,
                   device_type,
                   running_round,
                   restart_round,
                   limit_opencl_kernel_time,
                   opencl_queue_window_size,
                   tuning,
                   out_of_range_check,
                   model_graph_format,
                   opencl_binary_file,
                   opencl_parameter_file,
                   libmace_dynamic_library_path,
                   num_threads=-1,
                   cpu_affinity_policy=1,
                   cl_cache_reuse_policy=1,
                   gpu_perf_hint=3,
                   gpu_priority_hint=3,
                   accelerator_cache_policy=0,
                   accelerator_binary_file="",
                   accelerator_storage_file="",
                   apu_boost_hint=100,
                   apu_preference_hint=1,
                   input_file_name='model_input',
                   output_file_name='model_out',
                   input_dir="",
                   output_dir="",
                   runtime_failure_ratio=0.0,
                   address_sanitizer=False,
                   link_dynamic=False,
                   quantize_stat=False,
                   layers_validate_file="",
                   benchmark=False,
                   fake_warmup=False,
                   use_system_libhexagon_nn=False,
                   debug_mode=False,
                   ):
        six.print_("* Run '%s' with round=%s, restart_round=%s, tuning=%s, "
                   "out_of_range_check=%s, num_threads=%s, "
                   "cpu_affinity_policy=%s, gpu_perf_hint=%s, "
                   "gpu_priority_hint=%s" %
                   (model_tag, running_round, restart_round, str(tuning),
                    str(out_of_range_check), num_threads,
                    cpu_affinity_policy, gpu_perf_hint, gpu_priority_hint))
        mace_model_path = ""
        if model_graph_format == ModelFormat.file:
            mace_model_path = layers_validate_file if layers_validate_file \
                else "%s/%s.pb" % (mace_model_dir, model_tag)

        model_data_file = ""
        if not embed_model_data:
            if self.system == SystemType.host:
                model_data_file = "%s/%s.data" % (mace_model_dir, model_tag)
            else:
                model_data_file = "%s/%s.data" % (self.data_dir, model_tag)

        if self.system == SystemType.host:
            libmace_dynamic_lib_path = \
                os.path.dirname(libmace_dynamic_library_path)
            p = subprocess.Popen(
                [
                    "env",
                    "ASAN_OPTIONS=detect_leaks=1",
                    "LD_LIBRARY_PATH=%s" % libmace_dynamic_lib_path,
                    "MACE_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
                    "MACE_RUNTIME_FAILURE_RATIO=%f" % runtime_failure_ratio,
                    "MACE_LOG_TENSOR_RANGE=%d" % (1 if quantize_stat else 0),
                    "%s/%s" % (target_dir, target_name),
                    "--model_name=%s" % model_tag,
                    "--input_node=%s" % ",".join(input_nodes),
                    "--output_node=%s" % ",".join(output_nodes),
                    "--input_shape=%s" % ":".join(input_shapes),
                    "--output_shape=%s" % ":".join(output_shapes),
                    "--input_data_type=%s" % ",".join(input_data_types),
                    "--output_data_type=%s" % ",".join(output_data_types),
                    "--input_data_format=%s" % ",".join(input_data_formats),
                    "--output_data_format=%s" % ",".join(output_data_formats),
                    "--input_file=%s/%s" % (model_output_dir,
                                            input_file_name),
                    "--output_file=%s/%s" % (model_output_dir,
                                             output_file_name),
                    "--input_dir=%s" % input_dir,
                    "--output_dir=%s" % output_dir,
                    "--model_data_file=%s" % model_data_file,
                    "--round=%s" % running_round,
                    "--restart_round=%s" % restart_round,
                    "--num_threads=%s" % num_threads,
                    "--cpu_affinity_policy=%s" % cpu_affinity_policy,
                    "--opencl_cache_reuse_policy=%s" % cl_cache_reuse_policy,
                    "--gpu_perf_hint=%s" % gpu_perf_hint,
                    "--gpu_priority_hint=%s" % gpu_priority_hint,
                    "--model_file=%s" % mace_model_path,
                ],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True)
            out, err = p.communicate()
            self.stdout = err + out
            six.print_(self.stdout)
            mace_check(p.returncode == 0,
                       ModuleName.RUN,
                       "Failed to run the model")
            six.print_("Running finished!\n")
        elif self.system in [SystemType.android, SystemType.arm_linux]:
            self.rm(self.data_dir)
            self.exec_command('mkdir -p {}'.format(self.data_dir))
            internal_storage_dir = self.create_internal_storage_dir()

            for input_name in input_nodes:
                formatted_name = common.formatted_file_name(input_file_name,
                                                            input_name)
                self.push("%s/%s" % (model_output_dir, formatted_name),
                          self.data_dir)
            if self.system == SystemType.android and address_sanitizer:
                self.push(sh_commands.find_asan_rt_library(abi),
                          self.data_dir)

            if not embed_model_data:
                model_data_path = "%s/%s.data" % (mace_model_dir, model_tag)
                mace_check(os.path.exists(model_data_path), "Device",
                           'model data file not found,'
                           ' please convert model first')
                self.push(model_data_path, self.data_dir)

            if device_type == common.DeviceType.GPU:
                if os.path.exists(opencl_binary_file):
                    self.push(opencl_binary_file, self.interior_dir)
                if os.path.exists(opencl_parameter_file):
                    self.push(opencl_parameter_file, self.data_dir)

            if self.system == SystemType.android and \
                    device_type != common.DeviceType.APU:
                self.push(
                    "third_party/nnlib/%s/libhexagon_controller.so" % abi,
                    self.data_dir)

            cmd = [
                "LD_LIBRARY_PATH=%s" % self.data_dir,
                "MACE_TUNING=%s" % int(tuning),
                "MACE_OUT_OF_RANGE_CHECK=%s" % int(out_of_range_check),
                "MACE_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
                "MACE_RUN_PARAMETER_PATH=%s/mace_run.config" % self.data_dir,
                "MACE_INTERNAL_STORAGE_PATH=%s" % internal_storage_dir,
                "MACE_LIMIT_OPENCL_KERNEL_TIME=%s" % limit_opencl_kernel_time,
                "MACE_OPENCL_QUEUE_WINDOW_SIZE=%s" % opencl_queue_window_size,
                "MACE_RUNTIME_FAILURE_RATIO=%f" % runtime_failure_ratio,
                "MACE_LOG_TENSOR_RANGE=%d" % (1 if quantize_stat else 0),
            ]

            apu_storage_cpy = False
            if device_type == common.DeviceType.APU:
                if accelerator_cache_policy == 1:
                    if not accelerator_storage_file:
                        apu_storage_cpy = True
                        apu_src_file = model_tag + ".bin"
                        accelerator_storage_file = os.path.join(self.data_dir,
                                                                apu_src_file)
                elif accelerator_cache_policy in (2, 3):
                    if len(accelerator_binary_file) == 0:
                        apu_src_file = model_tag + ".bin"
                        accelerator_binary_file = '{}/apu_init_cache/{}'.format(mace_model_dir, apu_src_file)
                    if os.path.exists(accelerator_binary_file):
                        self.push(accelerator_binary_file, self.data_dir)
                        accelerator_binary_file = os.path.join(
                            self.data_dir, os.path.basename(accelerator_binary_file))
                for so_path in self.get_apu_so_paths():
                    self.push(so_path, self.data_dir)
                if debug_mode:
                    cmd.append("MTKNN_ENABLE_PROFILER=1")
            soc_id = 0
            if device_type in [common.DeviceType.HEXAGON, common.DeviceType.HTP]:
                ret = sh.adb(
                    "-s", self.address, "shell",
                    "if [ -f '/sys/devices/soc0/soc_id' ];"
                    "then cat /sys/devices/soc0/soc_id;"
                    "elif [ -f '/sys/devices/system/soc/soc0/id' ];"
                    "then cat /sys/devices/system/soc/soc0/id;"
                    "else echo -1; fi")
                soc_id = ret.stdout
                if len(soc_id) > 1 and soc_id[:-1].isdigit():
                    soc_id = int(soc_id[:-1])
                else:
                    mace_check(
                        False, ModuleName.RUN,
                        "The device do not have a valid hexagon dsp")
            if (device_type == common.DeviceType.HEXAGON and
                    not use_system_libhexagon_nn):
                cmd.append("ADSP_LIBRARY_PATH=%s" % self.data_dir)
                skel_info = get_soc_skel_info(soc_id)
                hexagon_skel = skel_info.skel.value
                print("The hexagon dsp skel : " + str(hexagon_skel))
                libhexgon_nn_skel_shared_lib = os.path.join(
                    "/tmp",
                    "libhexagon_nn_skel.so")
                ret = urllib.request.urlretrieve(
                    "https://cnbj1-fds.api.xiaomi.net/mace/third-party/nnlib/"
                    "%s/libhexagon_nn_skel.so" % hexagon_skel.lower(),
                    libhexgon_nn_skel_shared_lib)
                self.push(libhexgon_nn_skel_shared_lib, self.data_dir)
                sh.rm(libhexgon_nn_skel_shared_lib)
            if device_type == common.DeviceType.HTP:
                cmd.append("ADSP_LIBRARY_PATH=%s" % self.data_dir)
                skel_info = get_soc_skel_info(soc_id, DspType.HTP)
                qnn_skel = skel_info.skel.value
                qnn_abi = ""
                if "arm64-v8a" == abi:
                    qnn_abi = "aarch64-android"
                elif "armeabi-v7a" == abi:
                    qnn_abi = "arm-android"
                self.push(
                    "third_party/qnn/target/%s/lib/libQnnHtp%sStub.so" % (qnn_abi, qnn_skel),
                    self.data_dir)
                self.push(
                    "third_party/qnn/target/%s/lib/libQnnHtp.so" % qnn_abi,
                    self.data_dir)
                self.push(
                    "third_party/qnn/target/hexagon-%s/lib/unsigned/libQnnHtp%sSkel.so" %
                    (qnn_skel.lower(), qnn_skel),
                    self.data_dir)
                self.push(
                    "third_party/qnn/target/%s/lib/libQnnHtpPrepare.so" % qnn_abi,
                    self.data_dir)
                self.push(
                    "third_party/qnn/target/%s/lib/libQnnHtpNetRunExtensions.so" % qnn_abi,
                    self.data_dir)
            mace_model_phone_path = ""
            if model_graph_format == ModelFormat.file:
                mace_model_phone_path = "%s/%s.pb" % (self.data_dir,
                                                      model_tag)
                self.push(mace_model_path, mace_model_phone_path)
            if link_dynamic:
                self.push(libmace_dynamic_library_path, self.data_dir)
                if self.system == SystemType.android:
                    sh_commands.push_depended_so_libs(
                        libmace_dynamic_library_path, abi, self.data_dir,
                        self.address)
            self.push("%s/%s" % (target_dir, target_name), self.data_dir)

            stdout_buff = []
            process_output = sh_commands.make_output_processor(stdout_buff)

            if self.system == SystemType.android and address_sanitizer:
                cmd.extend([
                    "LD_PRELOAD=%s/%s" %
                    (self.data_dir,
                     sh_commands.asan_rt_library_names(abi))
                ])
            cmd.extend([
                "%s/%s" % (self.data_dir, target_name),
                "--model_name=%s" % model_tag,
                "--input_node='%s'" % ",".join(input_nodes),
                "--output_node='%s'" % ",".join(output_nodes),
                "--input_shape=%s" % ":".join(input_shapes),
                "--output_shape=%s" % ":".join(output_shapes),
                "--input_data_type=%s" % ",".join(input_data_types),
                "--output_data_type=%s" % ",".join(output_data_types),
                "--input_data_format=%s" % ",".join(input_data_formats),
                "--output_data_format=%s" % ",".join(output_data_formats),
                "--input_file=%s/%s" % (self.data_dir, input_file_name),
                "--output_file=%s/%s" % (self.data_dir, output_file_name),
                "--input_dir=%s" % input_dir,
                "--output_dir=%s" % output_dir,
                "--model_data_file=%s" % model_data_file,
                "--round=%s" % running_round,
                "--restart_round=%s" % restart_round,
                "--num_threads=%s" % num_threads,
                "--cpu_affinity_policy=%s" % cpu_affinity_policy,
                "--opencl_cache_reuse_policy=%s" % cl_cache_reuse_policy,
                "--opencl_cache_full_path=%s/%s" %
                (DEVICE_INTERIOR_DIR, CL_COMPILED_BINARY_FILE_NAME),
                "--gpu_perf_hint=%s" % gpu_perf_hint,
                "--gpu_priority_hint=%s" % gpu_priority_hint,
                "--model_file=%s" % mace_model_phone_path,
                "--opencl_binary_file=%s/%s" %
                (self.data_dir, os.path.basename(opencl_binary_file)),
                "--opencl_parameter_file=%s/%s" %
                (self.data_dir, os.path.basename(opencl_parameter_file)),
                "--accelerator_cache_policy=%s" % accelerator_cache_policy,
                "--accelerator_binary_file=%s" % accelerator_binary_file,
                "--accelerator_storage_file=%s" % accelerator_storage_file,
                "--apu_boost_hint=%s" % apu_boost_hint,
                "--apu_preference_hint=%s" % apu_preference_hint,
            ])
            if benchmark:
                cmd.append("--benchmark=%s" % benchmark)
            if fake_warmup:
                cmd.append("--fake_warmup=%s" % fake_warmup)

            cmd = ' '.join(cmd)
            cmd_file_name = "%s-%s-%s" % ('cmd_file',
                                          model_tag,
                                          str(time.time()))
            cmd_file = "%s/%s" % (self.data_dir, cmd_file_name)
            tmp_cmd_file = "%s/%s" % ('/tmp', cmd_file_name)
            with open(tmp_cmd_file, 'w') as file:
                file.write(cmd)
            self.push(tmp_cmd_file, cmd_file)
            os.remove(tmp_cmd_file)
            self.exec_command('sh {}'.format(cmd_file),
                              _tty_in=True,
                              _out=process_output,
                              _err_to_out=True)
            self.stdout = "".join(stdout_buff)

            if apu_storage_cpy:
                self.pull_from_data_dir(
                    apu_src_file, '{}/apu_init_cache/'.format(mace_model_dir))

            if not sh_commands.stdout_success(self.stdout):
                common.MaceLogger.error("Mace Run", "Mace run failed.")

            six.print_("Running finished!\n")
        else:
            six.print_('Unsupported system %s' % self.system, file=sys.stderr)
            raise Exception('Wrong device')

        return self.stdout

    def tuning(self, library_name, model_name, model_config,
               model_graph_format, model_data_format,
               target_abi, mace_lib_type):
        six.print_('* Tuning, it may take some time')
        build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
        mace_run_name = MACE_RUN_STATIC_NAME
        link_dynamic = False
        if mace_lib_type == MACELibType.dynamic:
            mace_run_name = MACE_RUN_DYNAMIC_NAME
            link_dynamic = True
        embed_model_data = model_data_format == ModelFormat.code

        # build for specified soc
        # device_wrapper = DeviceWrapper(device)

        model_output_base_dir, model_output_dir, mace_model_dir = \
            get_build_model_dirs(
                library_name, model_name, target_abi, self,
                model_config[YAMLKeyword.model_file_path])

        self.clear_data_dir()

        subgraphs = model_config[YAMLKeyword.subgraphs]
        # generate input data
        input_tensors_info = config_parser.find_input_tensors_info(
            subgraphs, model_config[YAMLKeyword.input_tensors])
        sh_commands.gen_input(
            model_output_dir,
            input_tensors_info[YAMLKeyword.input_tensors],
            input_tensors_info[YAMLKeyword.input_shapes],
            model_config[YAMLKeyword.validation_inputs_data],
            input_ranges=model_config[YAMLKeyword.input_ranges],
            input_data_types=input_tensors_info[YAMLKeyword.input_data_types]
        )

        output_tensors_info = config_parser.find_output_tensors_info(
            subgraphs, model_config[YAMLKeyword.output_tensors])
        self.tuning_run(
            abi=target_abi,
            target_dir=build_tmp_binary_dir,
            target_name=mace_run_name,
            vlog_level=0,
            embed_model_data=embed_model_data,
            model_output_dir=model_output_dir,
            input_nodes=input_tensors_info[YAMLKeyword.input_tensors],
            output_nodes=output_tensors_info[YAMLKeyword.output_tensors],
            input_shapes=input_tensors_info[YAMLKeyword.input_shapes],
            output_shapes=output_tensors_info[YAMLKeyword.output_shapes],
            input_data_types=input_tensors_info[YAMLKeyword.input_data_types],
            output_data_types=output_tensors_info[
                YAMLKeyword.output_data_types],
            input_data_formats=input_tensors_info[
                YAMLKeyword.input_data_formats],
            output_data_formats=output_tensors_info[
                YAMLKeyword.output_data_formats],
            mace_model_dir=mace_model_dir,
            model_tag=model_name,
            device_type=DeviceType.GPU,
            running_round=0,
            restart_round=1,
            limit_opencl_kernel_time=model_config[
                YAMLKeyword.limit_opencl_kernel_time],
            opencl_queue_window_size=model_config[
                YAMLKeyword.opencl_queue_window_size],
            tuning=True,
            out_of_range_check=False,
            model_graph_format=model_graph_format,
            opencl_binary_file='',
            opencl_parameter_file='',
            libmace_dynamic_library_path=LIBMACE_DYNAMIC_PATH,
            link_dynamic=link_dynamic,
        )

        # pull opencl library
        self.pull(self.interior_dir, CL_COMPILED_BINARY_FILE_NAME,
                  '{}/{}'.format(model_output_dir,
                                 BUILD_TMP_OPENCL_BIN_DIR))

        # pull opencl parameter
        self.pull_from_data_dir(CL_TUNED_PARAMETER_FILE_NAME,
                                '{}/{}'.format(model_output_dir,
                                               BUILD_TMP_OPENCL_BIN_DIR))

        six.print_('Tuning done! \n')

    @staticmethod
    def get_layers(model_dir, model_name, layers):
        model_file = "%s/%s.pb" % (model_dir, model_name)
        output_dir = "%s/output_models/" % model_dir
        if os.path.exists(output_dir):
            sh.rm('-rf', output_dir)
        os.makedirs(output_dir)

        layers_validate.convert(model_file, output_dir, layers)

        output_configs_path = output_dir + "outputs.yml"
        with open(output_configs_path) as f:
            output_configs = yaml.load(f)
        output_configs = output_configs[YAMLKeyword.subgraphs]

        return output_configs

    def get_input_output_infos(self, model_config):
        subgraphs = model_config[YAMLKeyword.subgraphs]
        if YAMLKeyword.input_tensors in model_config:
            input_tensors = model_config[YAMLKeyword.input_tensors]
            output_tensors = model_config[YAMLKeyword.output_tensors]
        else:
            mace_check(len(subgraphs.items()) == 1, "Device",
                       "You should specify input and output tensors in yml")
            for (graph_name, graph_config) in subgraphs.items():
                input_tensors = graph_config[YAMLKeyword.input_tensors]
                output_tensors = graph_config[YAMLKeyword.output_tensors]
                if YAMLKeyword.validation_inputs_data in graph_config:
                    model_config[YAMLKeyword.validation_inputs_data] = \
                        graph_config[YAMLKeyword.validation_inputs_data]
                if YAMLKeyword.validation_outputs_data in graph_config:
                    model_config[YAMLKeyword.validation_outputs_data] = \
                        graph_config[YAMLKeyword.validation_outputs_data]
                break

        input_infos = \
            config_parser.find_input_tensors_info(subgraphs, input_tensors)
        output_infos = config_parser.find_output_tensors_info(
            subgraphs, output_tensors)
        return (input_infos, output_infos)

    def run_model(self, flags, configs, target_abi,
                  model_name, output_config, runtime, tuning):
        library_name = configs[YAMLKeyword.library_name]
        embed_model_data = \
            configs[YAMLKeyword.model_data_format] == ModelFormat.code
        build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
        # get target name for run
        mace_lib_type = flags.mace_lib_type
        if mace_lib_type == MACELibType.static:
            target_name = MACE_RUN_STATIC_NAME
        else:
            target_name = MACE_RUN_DYNAMIC_NAME
        link_dynamic = mace_lib_type == MACELibType.dynamic

        if target_abi != ABIType.host:
            self.clear_data_dir()

        model_config = configs[YAMLKeyword.models][model_name]
        (input_infos, output_infos) = self.get_input_output_infos(model_config)

        model_output_base_dir, model_output_dir, mace_model_dir = \
            get_build_model_dirs(
                library_name, model_name, target_abi, self,
                model_config[YAMLKeyword.model_file_path])

        model_opencl_output_bin_path = ''
        model_opencl_parameter_path = ''
        if tuning:
            model_opencl_output_bin_path = \
                '{}/{}/{}'.format(model_output_dir,
                                  BUILD_TMP_OPENCL_BIN_DIR,
                                  CL_COMPILED_BINARY_FILE_NAME)
            model_opencl_parameter_path = \
                '{}/{}/{}'.format(model_output_dir,
                                  BUILD_TMP_OPENCL_BIN_DIR,
                                  CL_TUNED_PARAMETER_FILE_NAME)
        elif target_abi != ABIType.host and self.target_socs:
            model_opencl_output_bin_path = get_opencl_binary_output_path(
                library_name, target_abi, self
            )
            model_opencl_parameter_path = get_opencl_parameter_output_path(
                library_name, target_abi, self
            )
        # run for specified soc
        device_type = parse_device_type(runtime)
        self.tuning_run(
            abi=target_abi,
            target_dir=build_tmp_binary_dir,
            target_name=target_name,
            vlog_level=flags.vlog_level,
            embed_model_data=embed_model_data,
            model_output_dir=model_output_dir,
            input_nodes=input_infos[YAMLKeyword.input_tensors],
            output_nodes=output_config[YAMLKeyword.output_tensors],
            input_shapes=input_infos[YAMLKeyword.input_shapes],
            output_shapes=output_config[YAMLKeyword.output_shapes],
            input_data_types=input_infos[YAMLKeyword.input_data_types],
            output_data_types=output_infos[YAMLKeyword.output_data_types],
            input_data_formats=input_infos[YAMLKeyword.input_data_formats],
            output_data_formats=output_infos[YAMLKeyword.output_data_formats],
            mace_model_dir=mace_model_dir,
            model_tag=model_name,
            device_type=device_type,
            running_round=flags.round,
            restart_round=flags.restart_round,
            limit_opencl_kernel_time=model_config[
                YAMLKeyword.limit_opencl_kernel_time],
            opencl_queue_window_size=model_config[
                YAMLKeyword.opencl_queue_window_size],
            tuning=False,
            out_of_range_check=flags.gpu_out_of_range_check,
            model_graph_format=configs[
                YAMLKeyword.model_graph_format],
            num_threads=flags.num_threads,
            cpu_affinity_policy=flags.cpu_affinity_policy,
            cl_cache_reuse_policy=flags.opencl_cache_reuse_policy,
            gpu_perf_hint=flags.gpu_perf_hint,
            gpu_priority_hint=flags.gpu_priority_hint,
            accelerator_cache_policy=flags.accelerator_cache_policy,
            accelerator_binary_file=flags.accelerator_binary_file,
            accelerator_storage_file=flags.accelerator_storage_file,
            apu_boost_hint=flags.apu_boost_hint,
            apu_preference_hint=flags.apu_preference_hint,
            runtime_failure_ratio=flags.runtime_failure_ratio,
            address_sanitizer=flags.address_sanitizer,
            opencl_binary_file=model_opencl_output_bin_path,
            opencl_parameter_file=model_opencl_parameter_path,
            libmace_dynamic_library_path=LIBMACE_DYNAMIC_PATH,
            link_dynamic=link_dynamic,
            quantize_stat=flags.quantize_stat,
            input_dir=flags.input_dir,
            output_dir=flags.output_dir,
            layers_validate_file=output_config[
                YAMLKeyword.model_file_path],
            benchmark=flags.benchmark,
            fake_warmup=flags.fake_warmup,
            use_system_libhexagon_nn=flags.use_system_libhexagon_nn,
            debug_mode=flags.debug_mode,
        )

    def get_output_map(self,
                       target_abi,
                       output_nodes,
                       output_shapes,
                       model_output_dir):
        output_map = {}
        for i in range(len(output_nodes)):
            output_name = output_nodes[i]
            formatted_name = common.formatted_file_name(
                "model_out", output_name)
            if target_abi != "host":
                if os.path.exists("%s/%s" % (model_output_dir,
                                             formatted_name)):
                    sh.rm("-rf", "%s/%s" % (model_output_dir,
                                            formatted_name))
                self.pull_from_data_dir(formatted_name,
                                        model_output_dir)
            output_file_path = os.path.join(model_output_dir,
                                            formatted_name)
            output_shape = [
                int(x) for x in common.split_shape(output_shapes[i])]
            output_map[output_name] = np.fromfile(
                output_file_path, dtype=np.float32).reshape(output_shape)
        return output_map

    @staticmethod
    def get_graph_runtime(graph_config, model_config, target_abi):
        if target_abi == ABIType.host:
            graph_runtime = RuntimeType.cpu
        elif YAMLKeyword.runtime in graph_config:
            graph_runtime = graph_config[YAMLKeyword.runtime]
        else:
            graph_runtime = model_config[YAMLKeyword.runtime]
        if graph_runtime == RuntimeType.cpu_gpu:
            graph_runtime = RuntimeType.gpu
        return graph_runtime

    def run_specify_abi(self, flags, configs, target_abi):
        if target_abi not in self.target_abis:
            six.print_('The device %s with soc %s do not support the abi %s' %
                       (self.device_name, self.target_socs, target_abi))
            return
        library_name = configs[YAMLKeyword.library_name]

        model_output_dirs = []

        for model_name in configs[YAMLKeyword.models]:
            check_model_converted(library_name, model_name,
                                  configs[YAMLKeyword.model_graph_format],
                                  configs[YAMLKeyword.model_data_format],
                                  target_abi)
            if target_abi != ABIType.host:
                self.clear_data_dir()
            MaceLogger.header(
                StringFormatter.block(
                    'Run model {} on {}'.format(model_name, self.device_name)))

            model_config = configs[YAMLKeyword.models][model_name]
            if model_config[YAMLKeyword.platform] == 'pytorch':
                mace_check(flags.layers == "-1", "Device",
                           'extracting intermediate layer output is not supported in pytorch JIT yet')  # noqa

            runtime_list = []
            subgraphs = model_config[YAMLKeyword.subgraphs]
            for (graph_name, graph_config) in subgraphs.items():
                graph_runtime = self.get_graph_runtime(
                    graph_config, model_config, target_abi)
                runtime_list.append(graph_runtime)
                mace_check(graph_runtime != RuntimeType.apu or
                           target_abi == ABIType.arm64_v8a, "Device",
                           "APU runtime does only support arm64-v8a")

            model_output_base_dir, model_output_dir, mace_model_dir = \
                get_build_model_dirs(
                    library_name, model_name, target_abi, self,
                    model_config[YAMLKeyword.model_file_path])

            # clear temp model output dir
            if os.path.exists(model_output_dir):
                sh.rm('-rf', model_output_dir)
            os.makedirs(model_output_dir)

            tuning = False
            if not flags.address_sanitizer \
                    and target_abi != ABIType.host \
                    and (configs[YAMLKeyword.target_socs]
                         or flags.target_socs) \
                    and self.target_socs \
                    and RuntimeType.gpu in runtime_list \
                    and not flags.disable_tuning:
                self.tuning(library_name, model_name, model_config,
                            configs[YAMLKeyword.model_graph_format],
                            configs[YAMLKeyword.model_data_format],
                            target_abi, flags.mace_lib_type)
                model_output_dirs.append(model_output_dir)
                self.clear_data_dir()
                tuning = True

            (input_infos, output_infos) = \
                self.get_input_output_infos(model_config)
            accuracy_validation_script = \
                model_config[YAMLKeyword.accuracy_validation_script]
            output_configs = []
            if not accuracy_validation_script and flags.layers != "-1":
                mace_check(configs[YAMLKeyword.model_graph_format] ==
                           ModelFormat.file and
                           configs[YAMLKeyword.model_data_format] ==
                           ModelFormat.file, "Device",
                           "'--layers' only supports model format 'file'.")
                output_configs = self.get_layers(mace_model_dir,
                                                 model_name,
                                                 flags.layers)
            # run for specified soc
            output_nodes = output_infos[YAMLKeyword.output_tensors]
            output_shapes = output_infos[YAMLKeyword.output_shapes]
            if YAMLKeyword.check_tensors in output_infos:
                check_tensors = output_infos[YAMLKeyword.check_tensors]
                check_shapes = output_infos[YAMLKeyword.check_shapes]
                for i in range(len(check_tensors)):
                    if check_tensors[i]:
                        output_nodes[i] = check_tensors[i]
                        output_shapes[i] = check_shapes[i]

            model_path = "%s/%s.pb" % (mace_model_dir, model_name)
            output_config = {YAMLKeyword.model_file_path: model_path,
                             YAMLKeyword.output_tensors: output_nodes,
                             YAMLKeyword.output_shapes: output_shapes,
                             YAMLKeyword.output_data_formats:
                                 output_infos[YAMLKeyword.output_data_formats]}
            output_configs.append(output_config)

            if accuracy_validation_script:
                flags.validate = False
                flags.report = False

                import imp
                accuracy_val_module = imp.load_source(
                    'accuracy_val_module',
                    accuracy_validation_script)
                accuracy_validator = \
                    accuracy_val_module.AccuracyValidator()
                sample_size = accuracy_validator.sample_size()
                val_batch_size = accuracy_validator.batch_size()
                for i in range(0, sample_size, val_batch_size):
                    inputs = accuracy_validator.preprocess(
                        i, i + val_batch_size)
                    sh_commands.gen_input(
                        model_output_dir,
                        input_infos[YAMLKeyword.input_tensors],
                        input_infos[YAMLKeyword.input_shapes],
                        input_ranges=input_infos[YAMLKeyword.input_ranges],
                        input_data_types=input_infos[
                            YAMLKeyword.input_data_types],
                        input_data_map=inputs)

                    self.run_model(flags, configs, target_abi, model_name,
                                   output_configs[-1], runtime_list, tuning)
                    accuracy_validator.postprocess(
                        i, i + val_batch_size,
                        self.get_output_map(
                            target_abi,
                            output_nodes,
                            output_infos[YAMLKeyword.output_shapes],
                            model_output_dir))
                accuracy_validator.result()
            else:
                sh_commands.gen_input(
                    model_output_dir,
                    input_infos[YAMLKeyword.input_tensors],
                    input_infos[YAMLKeyword.input_shapes],
                    model_config[YAMLKeyword.validation_inputs_data],
                    input_ranges=input_infos[YAMLKeyword.input_ranges],
                    input_data_types=input_infos[YAMLKeyword.input_data_types]
                )

                # TODO(luxuhui): validate with the graph runtime
                runtime = runtime_list[0]
                device_type = parse_device_type(runtime)
                log_dir = mace_model_dir + "/" + runtime
                if os.path.exists(log_dir):
                    sh.rm('-rf', log_dir)
                os.makedirs(log_dir)
                for output_config in output_configs:
                    self.run_model(flags, configs, target_abi, model_name,
                                   output_config, runtime, tuning)
                    if flags.validate:
                        log_file = ""
                        if YAMLKeyword.output_data_types in output_config:
                            output_data_types = output_config[
                                YAMLKeyword.output_data_types]
                        else:
                            output_data_types = output_infos[
                                YAMLKeyword.output_data_types]
                        if flags.layers != "-1":
                            log_file = log_dir + "/log.csv"
                        model_file_path, weight_file_path = \
                            get_model_files(
                                model_config[YAMLKeyword.model_file_path],
                                model_config[
                                    YAMLKeyword.model_sha256_checksum],
                                BUILD_DOWNLOADS_DIR,
                                model_config[YAMLKeyword.weight_file_path],
                                model_config[
                                    YAMLKeyword.weight_sha256_checksum])
                        validate_type = device_type
                        if device_type in [DeviceType.CPU,
                                           DeviceType.GPU] and \
                                (model_config[YAMLKeyword.quantize] == 1 or
                                 model_config[
                                     YAMLKeyword.quantize_large_weights] == 1):  # noqa
                            validate_type = DeviceType.QUANTIZE

                        dockerfile_path, docker_image_tag = \
                            get_dockerfile_info(
                                model_config.get(
                                    YAMLKeyword.dockerfile_path),
                                model_config.get(
                                    YAMLKeyword.dockerfile_sha256_checksum),
                                # noqa
                                model_config.get(
                                    YAMLKeyword.docker_image_tag)
                            ) if YAMLKeyword.dockerfile_path \
                                 in model_config \
                                else ("third_party/caffe", "lastest")
                        sh_commands.validate_model(
                            abi=target_abi,
                            device=self,
                            model_file_path=model_file_path,
                            weight_file_path=weight_file_path,
                            docker_image_tag=docker_image_tag,
                            dockerfile_path=dockerfile_path,
                            platform=model_config[YAMLKeyword.platform],
                            device_type=device_type,
                            input_nodes=input_infos[
                                YAMLKeyword.input_tensors],
                            output_nodes=output_config[
                                YAMLKeyword.output_tensors],
                            input_shapes=input_infos[
                                YAMLKeyword.input_shapes],
                            output_shapes=output_config[
                                YAMLKeyword.output_shapes],
                            input_data_formats=input_infos[
                                YAMLKeyword.input_data_formats],
                            output_data_formats=output_config[
                                YAMLKeyword.output_data_formats],
                            model_output_dir=model_output_dir,
                            input_data_types=input_infos[
                                YAMLKeyword.input_data_types],
                            output_data_types=output_data_types,
                            caffe_env=flags.caffe_env,
                            validation_threshold=model_config[
                                YAMLKeyword.validation_threshold][
                                validate_type],
                            backend=model_config[YAMLKeyword.backend],
                            # TODO()
                            validation_outputs_data=model_config[
                                YAMLKeyword.validation_outputs_data],
                            log_file=log_file,
                        )
                    if flags.round > 0:
                        tuned = tuning and device_type == DeviceType.GPU
                        self.report_run_statistics(
                            target_abi=target_abi,
                            model_name=model_name,
                            device_type=device_type,
                            flags=flags,
                            tuned=tuned)

        if model_output_dirs:
            opencl_output_bin_path = get_opencl_binary_output_path(
                library_name, target_abi, self
            )
            opencl_parameter_bin_path = get_opencl_parameter_output_path(
                library_name, target_abi, self
            )

            # clear opencl output dir
            if os.path.exists(opencl_output_bin_path):
                sh.rm('-rf', opencl_output_bin_path)
            if os.path.exists(opencl_parameter_bin_path):
                sh.rm('-rf', opencl_parameter_bin_path)

            # merge all model's opencl binaries together
            sh_commands.merge_opencl_binaries_or_parameters(
                model_output_dirs, CL_COMPILED_BINARY_FILE_NAME,
                opencl_output_bin_path, sh_commands.MergeType.binary
            )
            # merge all model's opencl parameter together
            sh_commands.merge_opencl_binaries_or_parameters(
                model_output_dirs, CL_TUNED_PARAMETER_FILE_NAME,
                opencl_parameter_bin_path, sh_commands.MergeType.parameter
            )
            sh_commands.gen_opencl_binary_cpps(
                opencl_output_bin_path,
                opencl_parameter_bin_path,
                opencl_output_bin_path + '.cc',
                opencl_parameter_bin_path + '.cc')

    def report_run_statistics(self,
                              target_abi,
                              model_name,
                              device_type,
                              flags,
                              tuned):
        metrics = [0] * 3
        for line in self.stdout.split('\n'):
            line = line.strip()
            parts = line.split()
            if len(parts) == 5 and parts[0].startswith('time'):
                metrics[0] = str(float(parts[2]))
                metrics[1] = str(float(parts[3]))
                metrics[2] = str(float(parts[4]))
                break
        if flags.report:
            report_filename = flags.report_dir + '/report.csv'
            if not os.path.exists(report_filename):
                with open(report_filename, 'w') as f:
                    f.write('model_name,device_name,soc,abi,runtime,'
                            'init(ms),warmup(ms),run_avg(ms),tuned\n')

            data_str = \
                '{model_name},{device_name},{soc},{abi},{device_type},' \
                '{init},{warmup},{run_avg},{tuned}\n'.format(
                    model_name=model_name,
                    device_name=self.device_name,
                    soc=self.target_socs,
                    abi=target_abi,
                    device_type=device_type,
                    init=metrics[0],
                    warmup=metrics[1],
                    run_avg=metrics[2],
                    tuned=tuned)
            with open(report_filename, 'a') as f:
                f.write(data_str)
        self.report_to_dana(model_name, target_abi, device_type,
                            {
                                'init': metrics[0],
                                'warmup': metrics[1],
                                'run-avg': metrics[2]
                            })

    def report_to_dana(self, model_name, target_abi, device_type, metrics):
        if not self._dana_util.service_available():
            return
        for (key, value) in metrics.items():
            value = float(value)
            self._dana_util.fast_report_benchmark(
                table_name="Models", base_name="%s_%s" % (key, model_name),
                device_name=self.device_name, soc=self.target_socs,
                target_abi=target_abi, runtime=device_type,
                value=value, trend=DanaTrend.SMALLER)

    def run(self,
            abi,
            host_bin_path,
            bin_name,
            args='',
            opencl_profiling=True,
            vlog_level=0,
            out_of_range_check=True,
            address_sanitizer=False,
            simpleperf=False):
        host_bin_full_path = '%s/%s' % (host_bin_path, bin_name)
        device_bin_full_path = '%s/%s' % (self.data_dir, bin_name)
        print(
            '================================================================'
        )
        print('Trying to lock device %s' % self.address)
        with self.lock():
            print('Run on device: %s, %s, %s' %
                  (self.address, self.target_socs, self.device_name))
            self.rm(self.data_dir)
            self.exec_command('mkdir -p %s' % self.data_dir)
            self.push(host_bin_full_path, device_bin_full_path)
            ld_preload = ''
            if address_sanitizer:
                self.push(sh_commands.find_asan_rt_library(abi),
                          self.data_dir)
                ld_preload = 'LD_PRELOAD=%s/%s' % \
                             (self.data_dir,
                              sh_commands.asan_rt_library_names(abi))
            opencl_profiling = 1 if opencl_profiling else 0
            out_of_range_check = 1 if out_of_range_check else 0
            print('Run %s' % device_bin_full_path)
            stdout_buf = []
            process_output = sh_commands.make_output_processor(stdout_buf)

            internal_storage_dir = self.create_internal_storage_dir()

            if simpleperf and self.system == SystemType.android:
                self.push(sh_commands.find_simpleperf_library(abi),
                          self.data_dir)
                simpleperf_cmd = '%s/simpleperf' % self.data_dir
                exec_cmd = [
                    ld_preload,
                    'MACE_OUT_OF_RANGE_CHECK=%s' % out_of_range_check,
                    'MACE_OPENCL_PROFILING=%d' % opencl_profiling,
                    'MACE_INTERNAL_STORAGE_PATH=%s' % internal_storage_dir,
                    'MACE_CPP_MIN_VLOG_LEVEL=%d' % vlog_level,
                    simpleperf_cmd,
                    'stat',
                    '--group',
                    'raw-l1-dcache,raw-l1-dcache-refill',
                    '--group',
                    'raw-l2-dcache,raw-l2-dcache-refill',
                    '--group',
                    'raw-l1-dtlb,raw-l1-dtlb-refill',
                    '--group',
                    'raw-l2-dtlb,raw-l2-dtlb-refill',
                    device_bin_full_path,
                    args,
                ]
            else:
                exec_cmd = [
                    ld_preload,
                    'MACE_OUT_OF_RANGE_CHECK=%d' % out_of_range_check,
                    'MACE_OPENCL_PROFILNG=%d' % opencl_profiling,
                    'MACE_INTERNAL_STORAGE_PATH=%s' % internal_storage_dir,
                    'MACE_CPP_MIN_VLOG_LEVEL=%d' % vlog_level,
                    device_bin_full_path,
                    args
                ]
            exec_cmd = ' '.join(exec_cmd)
            self.exec_command(exec_cmd, _tty_in=True,
                              _out=process_output, _err_to_out=True)
            return ''.join(stdout_buf)


class DeviceManager:
    @classmethod
    def list_adb_device(cls):
        devices = []
        adb_list = sh_commands.adb_devices()
        for adb in adb_list:
            prop = sh_commands.adb_getprop_by_serialno(adb)
            android = {
                YAMLKeyword.device_name:
                    prop['ro.product.model'].replace(' ', ''),
                YAMLKeyword.target_abis:
                    prop['ro.product.cpu.abilist'].split(','),
                YAMLKeyword.target_socs: prop['ro.board.platform'],
                YAMLKeyword.system: SystemType.android,
                YAMLKeyword.address: adb,
                YAMLKeyword.username: '',
            }
            if android not in devices:
                devices.append(android)
        return devices

    @classmethod
    def list_ssh_device(cls, yml):
        with open(yml) as f:
            devices = yaml.load(f.read())
        devices = devices['devices']
        device_list = []
        for name, dev in six.iteritems(devices):
            dev[YAMLKeyword.device_name] = \
                dev[YAMLKeyword.models].replace(' ', '')
            dev[YAMLKeyword.system] = SystemType.arm_linux
            device_list.append(dev)
        return device_list

    @classmethod
    def list_devices(cls, yml):
        devices_list = []
        devices_list.extend(cls.list_adb_device())
        if not yml:
            if os.path.exists('devices.yml'):
                devices_list.extend(cls.list_ssh_device('devices.yml'))
        else:
            if os.path.exists(yml):
                devices_list.extend(cls.list_ssh_device(yml))
            else:
                MaceLogger.error(ModuleName.RUN,
                                 'no ARM linux device config file found')
        host = {
            YAMLKeyword.device_name: SystemType.host,
            YAMLKeyword.target_abis: [ABIType.host],
            YAMLKeyword.target_socs: '',
            YAMLKeyword.system: SystemType.host,
            YAMLKeyword.address: SystemType.host,

        }
        devices_list.append(host)
        return devices_list


if __name__ == '__main__':
    pass
