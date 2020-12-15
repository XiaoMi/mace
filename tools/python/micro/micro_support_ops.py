# Copyright 2020 The MACE Authors. All Rights Reserved.
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

import copy
from enum import Enum
from py_proto import mace_pb2
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from utils.config_parser import DataFormat
from utils.config_parser import ModelKeys
from utils.config_parser import Platform
from utils.util import mace_check
from micro.micro_op_resolver_rules import RefOPSResolverRules
from micro.micro_op_resolver_rules import OptOPSResolverRules
from micro.micro_op_resolver_cmsis_rules import CmsisOPSResolverRules
from micro.micro_op_resolver_xtensa_rules import XtensaOPSResolverRules


class OpResolver:
    def __init__(self, pb_model, model_conf):
        self.net_def = pb_model
        self.op_desc_map = {}
        self.op_desc_list = []

        self.backend = None
        if "micro" in model_conf:
            if "backend" in model_conf["micro"]:
                self.backend = model_conf["micro"]["backend"]

    def get_op_desc_list_from_model(self):
        op_class_name_list = []
        op_header_path_set = set()

        op_rules = []
        op_rules.extend(RefOPSResolverRules)
        op_rules.extend(OptOPSResolverRules)

        scratch_buffer_size = 0

        if self.backend == "cmsis":
            op_rules.extend(CmsisOPSResolverRules)

        if self.backend == "xtensa":
            op_rules.extend(XtensaOPSResolverRules)

        for op_def in self.net_def.op:
            cur_rule = None
            cur_priority = -1
            for rule in op_rules:
                valid = rule.valid(op_def, self.net_def)
                priority = rule.priority(op_def, self.net_def)
                if valid and priority > cur_priority:
                    cur_rule = rule
                    cur_priority = priority

            mace_check(
                cur_rule is not None,
                "unsupported op type %s." % op_def.type
            )

            cur_scratch_buffer_size = cur_rule.scratch(op_def, self.net_def)
            mace_check(cur_scratch_buffer_size >= 0,
                       "scratch buffer size must be ge than 0")
            scratch_buffer_size = int(max(
                scratch_buffer_size, cur_scratch_buffer_size))

            op_class_name_list.append(
                cur_rule.class_name(op_def, self.net_def))
            op_header_path_set.add(cur_rule.header_path(op_def, self.net_def))

        # 64 bytes is used for ignored small scratch bufffer
        scratch_buffer_size += 64
        op_header_path_list = list(op_header_path_set)

        return op_header_path_list, op_class_name_list, scratch_buffer_size
