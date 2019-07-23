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
import subprocess


MACE_TOOL_QUIET_ENV = "MACE_TOOL_QUIET"


def execute(cmd):
    print("CMD> %s" % cmd)
    p = subprocess.Popen([cmd],
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         stdin=subprocess.PIPE,
                         universal_newlines=True)
    returncode = p.poll()
    buf = []
    while returncode is None:
        line = p.stdout.readline()
        returncode = p.poll()
        line = line.strip()
        if MACE_TOOL_QUIET_ENV not in os.environ:
            print(line)
        buf.append(line)

    p.wait()

    if returncode != 0:
        raise Exception("errorcode: %s" % returncode)

    return "\n".join(buf)


class Device(object):
    def __init__(self, device_id):
        self._device_id = device_id

    def install(self, target, install_dir):
        pass

    def run(self, target):
        pass

    def pull(self, target, out_dir):
        pass


class HostDevice(Device):
    def __init__(self, device_id):
        super(HostDevice, self).__init__(device_id)

    @staticmethod
    def list_devices():
        return ["host"]

    def install(self, target, install_dir):
        install_dir = os.path.abspath(install_dir)

        if install_dir.strip() and install_dir != os.path.dirname(target.path):
            execute("mkdir -p %s" % install_dir)
            execute("cp %s %s" % (target.path, install_dir))
            for lib in target.libs:
                execute("cp %s %s" % (lib, install_dir))

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
            execute("cp -r %s %s" % (target.path, out_dir))


class AndroidDevice(Device):
    def __init__(self, device_id):
        super(AndroidDevice, self).__init__(device_id)

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

    def install(self, target, install_dir):
        install_dir = os.path.abspath(install_dir)
        sn = self._device_id

        execute("adb -s %s shell mkdir -p %s" % (sn, install_dir))
        execute("adb -s %s push %s %s" % (sn, target.path, install_dir))
        for lib in target.libs:
            execute("adb -s %s push %s %s" % (sn, lib, install_dir))

        target.path = "%s/%s" % (install_dir, os.path.basename(target.path))
        target.libs = ["%s/%s" % (install_dir, os.path.basename(lib))
                       for lib in target.libs]
        target.envs.append("LD_LIBRARY_PATH=%s" % install_dir)

        return target

    def run(self, target):
        out = execute("adb -s %s shell %s" % (self._device_id, target))
        # May have false positive using the following error word
        for line in out.split("\n")[:-10]:
            if ("Aborted" in line
                    or "FAILED" in line or "Segmentation fault" in line):
                raise Exception(line)

    def pull(self, target, out_dir):
        sn = self._device_id
        execute("adb -s %s pull %s %s" % (sn, target.path, out_dir))


class ArmLinuxDevice(Device):
    devices = {}

    def __init__(self, device_id):
        super(ArmLinuxDevice, self).__init__(device_id)

    @staticmethod
    def list_devices():
        device_ids = []
        for dev_name, dev_info in ArmLinuxDevice.devices:
            address = dev_info["address"]
            username = dev_info["username"]
            device_ids.append("%s@%s" % (username, address))

    @staticmethod
    def set_devices(devices):
        ArmLinuxDevice.devices = devices

    def install(self, target, install_dir):
        install_dir = os.path.abspath(install_dir)
        ip = self._device_id

        execute("ssh %s mkdir -p %s" % install_dir)
        execute("scp %s %s:%s" % (target.path, ip, install_dir))
        for lib in target.libs:
            execute("scp %s:%s" % (lib, install_dir))

        target.path = "%s/%s" % (install_dir, os.path.basename(target.path))
        target.libs = ["%s/%s" % (install_dir, os.path.basename(lib))
                       for lib in target.libs]
        target.envs.append("LD_LIBRARY_PATH=%s" % install_dir)

        return target

    def run(self, target):
        execute("ssh %s shell %s" % (self._device_id, target))

    def pull(self, target, out_dir):
        sn = self._device_id
        execute("scp %s:%s %s" % (sn, target.path, out_dir))


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


def crete_device(target_abi, device_id=None):
    return device_class(target_abi)(device_id)
