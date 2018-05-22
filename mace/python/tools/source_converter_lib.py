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

import datetime
import os

from mace.proto import mace_pb2
from jinja2 import Environment, FileSystemLoader


def stringfy(value):
    return ', '.join('"{0}"'.format(w) for w in value)


def convert_to_source(net_def, model_checksum, weight_checksum, template_dir,
                      obfuscate, model_tag, output, runtime, embed_model_data,
                      winograd_conv, model_load_type, tensor_infos,
                      model_data):

    # Capture our current directory
    print template_dir

    # Create the jinja2 environment.
    j2_env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True)
    j2_env.filters['stringfy'] = stringfy
    output_dir = os.path.dirname(output) + '/'
    # generate tensor source files
    template_name = 'tensor_source.jinja2'
    for i in range(len(net_def.tensors)):
        if model_load_type == 'source':
            source = j2_env.get_template(template_name).render(
                tensor_info=tensor_infos[i],
                tensor=net_def.tensors[i],
                tag=model_tag,
            )
            with open(output_dir + 'tensor' + str(i) + '.cc', "wb") as f:
                f.write(source)

    if model_load_type == 'source':
        # generate tensor data
        template_name = 'tensor_data.jinja2'
        source = j2_env.get_template(template_name).render(
            tag=model_tag,
            embed_model_data=embed_model_data,
            model_data_size=len(model_data),
            model_data=model_data)
        with open(output_dir + 'tensor_data' + '.cc', "wb") as f:
            f.write(source)

        # generate op source files
        template_name = 'operator.jinja2'
        counter = 0
        op_size = len(net_def.op)
        for start in range(0, op_size, 10):
            source = j2_env.get_template(template_name).render(
                start=start,
                end=min(start + 10, op_size),
                net=net_def,
                tag=model_tag,
                runtime=runtime,
            )
            with open(output_dir + 'op' + str(counter) + '.cc', "wb") as f:
                f.write(source)
            counter += 1

        # generate model source files
        build_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        template_name = 'model.jinja2'
        checksum = model_checksum
        if weight_checksum is not None:
            checksum = "{},{}".format(model_checksum, weight_checksum)
        source = j2_env.get_template(template_name).render(
            net=net_def,
            tag=model_tag,
            runtime=runtime,
            obfuscate=obfuscate,
            embed_model_data=embed_model_data,
            winograd_conv=winograd_conv,
            checksum=checksum,
            build_time=build_time)
        with open(output, "wb") as f:
            f.write(source)

        # generate model header file
        template_name = 'model_header.jinja2'
        source = j2_env.get_template(template_name).render(tag=model_tag, )
        with open(output_dir + model_tag + '.h', "wb") as f:
            f.write(source)
