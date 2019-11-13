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

import os
import shutil
import numpy as np

from micro.graph_builder import GraphBuilder
from micro.mem_computer import MemComputer
from micro.micro_codegen import MicroCodeGen
from micro.micro_io_converter import MicroIoConverter
from micro.micro_op_converter import MicroOpConverter
from micro.micro_support_ops import OpResolver
from micro.micro_support_ops import McSupportedOps
from micro.proto_to_bytes import ProtoConverter
from micro.scratch_computer import ScratchComputer
from py_proto import mace_pb2
from utils import util
from utils.config_parser import ModelKeys
from utils.convert_util import data_type_to_np_dt
from utils.util import mace_check

NetDefExcludeFields = {
    'OperatorDef': [
        'quantize_info',
        'node_id',
        'op_id',
        'padding',
        'node_input',
        'out_max_byte_size',
    ],
}


class MicroConverter:
    def __init__(self, model_conf, net_def, model_weights,
                 model_name, offset16=False, write_magic=False):
        self.model_conf = model_conf
        data_type = model_conf.get(ModelKeys.data_type, mace_pb2.DT_FLOAT)
        self.net_def = MicroIoConverter.convert(net_def, data_type)
        self.model_weights = model_weights
        self.model_name = model_name
        self.offset16 = offset16
        self.write_magic = write_magic
        self.code_gen = MicroCodeGen()
        data_type = model_conf.get(ModelKeys.data_type, mace_pb2.DT_FLOAT)
        self.np_data_type = data_type_to_np_dt(data_type, np.float32)
        self.gen_folder = 'micro/codegen/'
        util.mkdir_p(self.gen_folder)
        self.op_resolver = OpResolver(self.net_def, self.model_conf)

    def gen_code_from_model(self, model_name, pb_model, model_weights):
        net_def = pb_model
        output_dir = self.gen_folder + 'models/' + model_name + '/'
        shutil.rmtree(output_dir, ignore_errors=True)
        util.mkdir_p(output_dir)

        # comput mem size and mem block offset and update the net_def,
        # should count before ProtoConverter
        mem_computer = MemComputer(net_def, self.np_data_type)
        tensor_mem_size = mem_computer.compute()

        # gen the c++ NetDef struct
        net_def_converter = ProtoConverter(self.offset16, self.write_magic,
                                           NetDefExcludeFields)
        net_def_bytes = net_def_converter.proto_to_bytes(net_def)
        mace_check(net_def_bytes is not None, "proto_to_bytes failed.")
        self.code_gen.gen_net_def_data(model_name, net_def_bytes,
                                       output_dir + 'micro_net_def_data.h')

        # gen operator array
        (op_src_path_list, op_class_name_list) = \
            self.op_resolver.get_op_desc_list_from_model()
        self.code_gen.gen_ops_data(
            model_name, op_src_path_list, op_class_name_list,
            output_dir + 'micro_ops_list.h')

        # gen the c++ Graph struct
        graph = GraphBuilder(net_def, self.op_resolver).build()
        graph_converter = ProtoConverter(self.offset16, self.write_magic)
        graph_bytes = graph_converter.proto_to_bytes(graph)
        self.code_gen.gen_graph_data(model_name, graph_bytes,
                                     output_dir + 'micro_graph_data.h')

        scratch_buffer_size = ScratchComputer(
            net_def, self.model_conf).compute_size()

        # gen micro engine config
        engine_data = {}
        engine_data['tensor_mem_size'] = tensor_mem_size
        engine_data['input_size'] = len(net_def.input_info)
        engine_data['scratch_buffer_size'] = scratch_buffer_size
        self.code_gen.gen_engin_config(model_name, engine_data,
                                       output_dir + 'micro_engine_config.cc')

        # gen micro model tensor data
        tensor_bytes = bytearray(model_weights)
        self.code_gen.gen_model_data(model_name, tensor_bytes,
                                     output_dir + 'micro_model_data.h')

    def gen_engine_interface_code(self, model_name):
        output_dir = self.gen_folder + 'engines/' + model_name + '/'
        shutil.rmtree(output_dir, ignore_errors=True)
        util.mkdir_p(output_dir)
        self.code_gen.gen_engine_factory(
            model_name,
            output_dir + 'micro_engine_factory.h',
            output_dir + 'micro_engine_factory.cc')
        self.code_gen.gen_engine_c_interface(
            model_name,
            output_dir + 'micro_engine_c_interface.h',
            output_dir + 'micro_engine_c_interface.cc')

    def gen_code(self):
        MicroOpConverter(self.net_def, self.model_weights,
                         self.np_data_type).convert_op_params()
        self.gen_code_from_model(
            self.model_name, self.net_def, self.model_weights)
        self.gen_engine_interface_code(self.model_name)

    def package(self, tar_package_path):
        (op_h_path_list, op_class_name_list) = \
            self.op_resolver.get_op_desc_list_from_model()
        all_op_header_list = [op_desc.src_path for op_desc in McSupportedOps]
        op_h_exclude_list = []
        for op_header in all_op_header_list:
            if op_header not in op_h_path_list:
                op_h_exclude_list.append(op_header)
        op_cc_exclude_list = \
            [op_h.replace(".h", ".cc") for op_h in op_h_exclude_list]
        exclude_list = ["--exclude=" + op_h for op_h in op_h_exclude_list]
        exclude_list.extend(
            ["--exclude=" + op_h for op_h in op_cc_exclude_list])
        tmp_dir = "/tmp/micro"
        tmp_workspace_file = "WORKSPACE"
        os.system("mkdir -p %s && touch %s/%s" %
                  (tmp_dir, tmp_dir, tmp_workspace_file))
        tar_command = "tar --exclude=micro/tools --exclude=micro/test "
        tar_command += " ".join(exclude_list)
        tar_command += " -zcf " + tar_package_path
        tar_command += " micro -C %s %s" % (tmp_dir, tmp_workspace_file)
        os.system(tar_command)
