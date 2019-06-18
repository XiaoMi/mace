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

import re
import os
import yaml


def sanitize_load(s):
    # do not let yaml parse ON/OFF to boolean
    for w in ["ON", "OFF", "on", "off"]:
        s = re.sub(r":\s+" + w, r": '" + w + "'", s)

    # sub ${} to env value
    s = re.sub(r"\${(\w+)}", lambda x: os.environ[x.group(1)], s)
    return yaml.load(s)


def parse(path):
    with open(path) as f:
        config = sanitize_load(f.read())

    return config


def parse_device_info(path):
    conf = parse(path)
    return conf["devices"]
