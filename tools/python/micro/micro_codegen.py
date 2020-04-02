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


import numpy as np
import os

from jinja2 import Environment, FileSystemLoader

JINJA2_DIR = './jinja2_files/'


class MicroCodeGen:
    def __init__(self):
        pass

    def gen_micro_ops_list_from_bytes(self, model_tag, op_src_path_list,
                                      op_class_name_list,
                                      jinja_file_name, output_path):
        cwd = os.path.dirname(__file__)
        j2_env = Environment(
            loader=FileSystemLoader(cwd), trim_blocks=True)

        template_name = JINJA2_DIR + jinja_file_name
        source = j2_env.get_template(template_name).render(
            model_tag=model_tag,
            op_src_path_list=op_src_path_list,
            op_class_name_list=op_class_name_list,
            op_class_name_list_size=len(op_class_name_list)
        )
        with open(output_path, "w") as f:
            f.write(source)

    def gen_micro_source_from_bytes(self, model_tag, embed_data,
                                    jinja_file_name, output_path):
        cwd = os.path.dirname(__file__)
        j2_env = Environment(
            loader=FileSystemLoader(cwd), trim_blocks=True)

        template_name = JINJA2_DIR + jinja_file_name
        source = j2_env.get_template(template_name).render(
            model_tag=model_tag,
            embed_data=embed_data,
            data_size=len(embed_data),
        )
        with open(output_path, "w") as f:
            f.write(source)

    def gen_net_def_data(self, model_tag, model_def_data, output_path):
        embed_data = np.frombuffer(model_def_data, dtype=np.uint8)
        self.gen_micro_source_from_bytes(
            model_tag, embed_data, 'micro_net_def.h.jinja2', output_path)

    def gen_graph_data(self, model_tag, graph_data, output_path):
        embed_data = np.frombuffer(graph_data, dtype=np.uint8)
        self.gen_micro_source_from_bytes(model_tag, embed_data,
                                         'micro_graph_data.h.jinja2',
                                         output_path)

    def gen_ops_data(self, model_tag, op_src_path_list,
                     op_class_name_list, output_path):
        self.gen_micro_ops_list_from_bytes(model_tag, op_src_path_list,
                                           op_class_name_list,
                                           'micro_ops_list.h.jinja2',
                                           output_path)

    def gen_engin_config(self, model_tag, config_data, output_path):
        self.gen_micro_source_from_bytes(model_tag, config_data,
                                         'micro_engine_config.cc.jinja2',
                                         output_path)

    def gen_model_data(self, model_tag, model_param_data, output_path):
        embed_data = np.frombuffer(model_param_data, dtype=np.uint8)
        self.gen_micro_source_from_bytes(model_tag, embed_data,
                                         'micro_model_data.h.jinja2',
                                         output_path)

    def gen_engine_factory(self, model_tag, output_path_h, output_path_cc):
        self.gen_micro_source_from_bytes(model_tag, '',
                                         'micro_engine_factory.h.jinja2',
                                         output_path_h)
        self.gen_micro_source_from_bytes(model_tag, '',
                                         'micro_engine_factory.cc.jinja2',
                                         output_path_cc)

    def gen_engine_c_interface(self, model_tag, output_path_h, output_path_cc):
        self.gen_micro_source_from_bytes(model_tag, '',
                                         'micro_engine_c_interface.h.jinja2',
                                         output_path_h)
        self.gen_micro_source_from_bytes(model_tag, '',
                                         'micro_engine_c_interface.cc.jinja2',
                                         output_path_cc)
