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

import copy
import os
import re
import subprocess
import random
import tempfile

from utils import util


def execute(cmd, verbose=True):
    print("CMD> %s" % cmd)
    p = subprocess.Popen([cmd],
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         stdin=subprocess.PIPE,
                         universal_newlines=True)

    if not verbose:
        if p.wait() != 0:
            raise Exception("errorcode: %s" % p.returncode)
        return p.stdout.read()

    buf = []

    while p.poll() is None:
        line = p.stdout.readline().strip()
        if verbose:
            print(line)
        buf.append(line)

    for l in p.stdout:
        line = l.strip()
        if verbose:
            print(line)
        buf.append(line)

    if p.returncode != 0:
        if verbose:
            print(line)
        raise Exception("errorcode: %s" % p.returncode)

    return "\n".join(buf)


class Device(object):
    def __init__(self, device_id, target_abi):
        self._device_id = device_id
        self._target_abi = target_abi

    def install(self, target, install_dir, install_deps=False):
        pass

    def run(self, target):
        pass

    def pull(self, target, out_dir):
        pass

    def mkdir(self, dirname):
        pass

    def info(self):
        pass


class HostDevice(Device):
    def __init__(self, device_id, target_abi):
        super(HostDevice, self).__init__(device_id, target_abi)

    @staticmethod
    def list_devices():
        return ["host"]

    def install(self, target, install_dir, install_deps=False):
        install_dir = os.path.abspath(install_dir)

        if install_dir.strip() and install_dir != os.path.dirname(target.path):
            execute("mkdir -p %s" % install_dir)
            if os.path.isdir(target.path):
                execute("cp -f %s/* %s" % (target.path, install_dir))
            else:
                execute("cp -f %s %s" % (target.path, install_dir))
            for lib in target.libs:
                execute("cp -f %s %s" % (lib, install_dir))

            target.path = "%s/%s" % (install_dir,
                                     os.path.basename(target.path))
            target.libs = ["%s/%s" % (install_dir, os.path.basename(lib))
                           for lib in target.libs]

        target.envs.append("LD_LIBRARY_PATH=%s" % install_dir)

        return target

    def run(self, target):
        execute(str(target))

    def pull(self, target, out_dir):
        out_dir = os.path.abspath(out_dir)

        if out_dir.strip() and out_dir != os.path.dirname(target.path):
            execute("cp -rp %s %s" % (target.path, out_dir))

    def mkdir(self, dirname):
        execute("mkdir -p %s" % dirname)


class AndroidDevice(Device):
    def __init__(self, device_id, target_abi):
        super(AndroidDevice, self).__init__(device_id, target_abi)

    @staticmethod
    def list_devices():
        out = execute("adb devices")
        serialno_list = out.strip().split('\n')[1:]
        serialno_list = [tuple(pair.split('\t')) for pair in serialno_list]
        devices = []
        for serialno in serialno_list:
            if not serialno[1].startswith("no permissions"):
                devices.append(serialno[0])

        return devices

    def install(self, target, install_dir, install_deps=False):
        install_dir = os.path.abspath(install_dir)
        sn = self._device_id

        execute("adb -s %s shell mkdir -p %s" % (sn, install_dir))
        if os.path.isdir(target.path):
            execute("adb -s %s push %s/* %s" % (sn, target.path, install_dir),
                    False)
        else:
            execute("adb -s %s push %s %s" % (sn, target.path, install_dir),
                    False)

        for lib in target.libs:
            execute("adb -s %s push %s %s" % (sn, lib, install_dir), False)

        device_target = copy.deepcopy(target)
        device_target.path = "%s/%s" % (install_dir,
                                        os.path.basename(target.path))
        device_target.libs = ["%s/%s" % (install_dir, os.path.basename(lib))
                              for lib in target.libs]
        device_target.envs.append("LD_LIBRARY_PATH=%s" % install_dir)

        if install_deps:
            self.install_common_libs_for_target(target, install_dir)

        return device_target

    def install_common_libs_for_target(self, target, install_dir):
        sn = self._device_id
        dep_so_libs = execute(os.environ["ANDROID_NDK_HOME"] + "/ndk-depends "
                              + target.path)
        lib_file = ""
        for dep in dep_so_libs.split("\n"):
            if dep == "libgnustl_shared.so":
                lib_file = "%s/sources/cxx-stl/gnu-libstdc++/4.9/libs/" \
                           "%s/libgnustl_shared.so" \
                           % (os.environ["ANDROID_NDK_HOME"], self._target_abi)
            elif dep == "libc++_shared.so":
                lib_file = "%s/sources/cxx-stl/llvm-libc++/libs/" \
                           "%s/libc++_shared.so" \
                           % (os.environ["ANDROID_NDK_HOME"], self._target_abi)

        if lib_file:
            execute("adb -s %s push %s %s" % (sn, lib_file, install_dir),
                    False)

    def run(self, target):
        tmpdirname = tempfile.mkdtemp()
        cmd_file_path = tmpdirname + "/cmd.sh"
        with open(cmd_file_path, "w") as cmd_file:
            cmd_file.write(str(target))
        target_dir = os.path.dirname(target.path)
        execute("adb -s %s push %s %s" % (self._device_id,
                                          cmd_file_path,
                                          target_dir))

        out = execute("adb -s %s shell sh %s" % (self._device_id,
                                                 target_dir + "/cmd.sh"))
        # May have false positive using the following error word
        for line in out.split("\n")[:-10]:
            if ("Aborted" in line
                    or "FAILED" in line or "Segmentation fault" in line):
                raise Exception(line)

    def pull(self, target, out_dir):
        sn = self._device_id
        execute("adb -s %s pull %s %s" % (sn, target.path, out_dir), False)

    def mkdir(self, dirname):
        sn = self._device_id
        execute("adb -s %s shell mkdir -p %s" % (sn, dirname))

    def info(self):
        sn = self._device_id
        output = execute("adb -s %s shell getprop" % sn, False)
        raw_props = output.split("\n")
        props = {}
        p = re.compile(r'\[(.+)\]: \[(.+)\]')
        for raw_prop in raw_props:
            m = p.match(raw_prop)
            if m:
                props[m.group(1)] = m.group(2)
        return props


class ArmLinuxDevice(Device):
    devices = {}

    def __init__(self, device_id, target_abi):
        super(ArmLinuxDevice, self).__init__(device_id, target_abi)

    @staticmethod
    def list_devices():
        device_ids = []
        print("!!!", ArmLinuxDevice.devices)
        for dev_name, dev_info in ArmLinuxDevice.devices.items():
            address = dev_info["address"]
            username = dev_info["username"]
            device_ids.append("%s@%s" % (username, address))
        return device_ids

    @staticmethod
    def set_devices(devices):
        ArmLinuxDevice.devices = devices

    def install(self, target, install_dir, install_deps=False):
        install_dir = os.path.abspath(install_dir)
        ip = self._device_id

        execute("ssh %s mkdir -p %s" % (ip, install_dir))
        execute("scp -r %s %s:%s" % (target.path, ip, install_dir))
        for lib in target.libs:
            execute("scp -r %s:%s" % (lib, install_dir))

        target.path = "%s/%s" % (install_dir, os.path.basename(target.path))
        target.libs = ["%s/%s" % (install_dir, os.path.basename(lib))
                       for lib in target.libs]
        target.envs.append("LD_LIBRARY_PATH=%s" % install_dir)

        return target

    def run(self, target):
        execute("ssh %s %s" % (self._device_id, target))

    def pull(self, target, out_dir):
        sn = self._device_id
        execute("scp -r %s:%s %s" % (sn, target.path, out_dir))

    def mkdir(self, dirname):
        sn = self._device_id
        execute("ssh %s mkdir -p %s" % (sn, dirname))


def device_class(target_abi):
    device_dispatch = {
        "host": "HostDevice",
        "armeabi-v7a": "AndroidDevice",
        "arm64-v8a": "AndroidDevice",
        "arm-linux-gnueabihf": "ArmLinuxDevice",
        "aarch64-linux-gnu": "ArmLinuxDevice"
    }

    if target_abi not in device_dispatch:
        raise ValueError(
            "target_abi should be one of %s" % device_dispatch.keys())

    return globals()[device_dispatch[target_abi]]


def create_device(target_abi, device_id=None):
    return device_class(target_abi)(device_id, target_abi)


def choose_devices(target_abi, target_ids):
    device_clazz = device_class(target_abi)
    devices = device_clazz.list_devices()

    if target_ids == "all":
        run_devices = devices
    elif target_ids == "random":
        unlocked_devices = [dev for dev in devices if
                            not util.is_device_locked(dev)]
        if unlocked_devices:
            run_devices = [random.choice(unlocked_devices)]
        else:
            run_devices = [random.choice(devices)]
    else:
        device_id_list = [dev.strip() for dev in target_ids.split(",")]
        run_devices = [dev for dev in device_id_list if dev in devices]

    return run_devices
