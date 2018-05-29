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

import enum
import logging
import re


################################
# log
################################
def init_logging():
    logger = logging.getLogger('MACE')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] [%(levelname)s]: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


################################
# Argument types
################################
class CaffeEnvType(enum.Enum):
    DOCKER = 0,
    LOCAL = 1,


################################
# common functions
################################
def formatted_file_name(input_file_name, input_name):
    res = input_file_name + '_'
    for c in input_name:
        res += c if c.isalnum() else '_'
    return res
