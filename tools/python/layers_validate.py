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

import argparse
import copy
import os
import sh
import yaml

from py_proto import mace_pb2
from transform.base_converter import ConverterUtil
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from transform.hexagon_converter import HexagonOp
from utils.util import mace_check


def normalize_op_name(name):
    return name.replace('/', '_').replace(':', '_')


def handle_index(start, end, layers):
    num_layers = end - start + 1
    if ':' in layers:
        start_index, end_index = layers.split(':')
        start_index = int(start_index) if start_index else 0
        end_index = int(end_index) if end_index else num_layers - 1
    else:
        start_index = int(layers)
        end_index = start_index + 1
    if start_index < 0:
        start_index += num_layers
    if end_index < 0:
        end_index += num_layers
    start_index += start
    end_index += start
    start_index = max(start, min(end - 1, start_index))
    end_index = max(start + 1, min(end, end_index))

    return start_index, end_index


def save_model_to_proto(net_def, model_name, output_dir):
    output_path = output_dir + "/" + model_name + ".pb"
    with open(output_path, "wb") as f:
        f.write(net_def.SerializeToString())
    with open(output_path + "_txt", "w") as f:
        f.write(str(net_def))
    return output_path


def convert(model_file, output_dir, layers):
    mace_check(os.path.isfile(model_file),
               "Input graph file '" + model_file + "' does not exist!")
    mace_check(os.path.isdir(output_dir),
               "Output directory '" + output_dir + "' does not exist!")
    net_def = mace_pb2.NetDef()
    with open(model_file, "rb") as f:
        net_def.ParseFromString(f.read())

    is_quantize = ConverterUtil.get_arg(
        net_def, MaceKeyword.mace_quantize_flag_arg_str)
    is_quantize = False if is_quantize is None else is_quantize.i == 1
    is_hexagon = False
    index = 0
    end_index = len(net_def.op)
    if is_quantize:
        while index < end_index:
            # omit op quantize
            if net_def.op[index].type == MaceOp.Quantize.name or \
                    net_def.op[index].type == \
                    HexagonOp.QuantizeINPUT_f_to_8.name or \
                    net_def.op[index].type == HexagonOp.INPUT.name:
                index += 1
            # omit op dequantize
            elif net_def.op[end_index - 1].type == MaceOp.Dequantize.name or \
                    net_def.op[end_index - 1].type == \
                    HexagonOp.DequantizeOUTPUT_8tof.name or \
                    net_def.op[end_index - 1].type == HexagonOp.OUTPUT.name:

                end_index -= 1
            else:
                break
        mace_check(0 < index < end_index < len(net_def.op),
                   "Wrong number of op quantize(%d) or dequantize(%d)." %
                   (index, len(net_def.op) - end_index))
        if net_def.op[-1].type == HexagonOp.DequantizeOUTPUT_8tof.name or \
                net_def.op[-1].type == HexagonOp.OUTPUT.name:
            is_hexagon = True

    index, end_index = handle_index(index, end_index, layers)

    data_format = net_def.output_info[0].data_format
    output_configs = {"subgraphs": []}
    while index < end_index:
        # omit BatchToSpaceND and op before that due to changed graph
        if net_def.op[index].type == MaceOp.BatchToSpaceND.name or \
                net_def.op[index].type == HexagonOp.BatchToSpaceND_8.name or \
                (index + 1 < end_index and
                 (net_def.op[index + 1].type == MaceOp.BatchToSpaceND.name or
                  net_def.op[index + 1].type == HexagonOp.BatchToSpaceND_8.name)):  # noqa
            index += 1
            continue
        net = copy.deepcopy(net_def)
        if is_hexagon:
            # reuse dequantize op and it's min/max tensor's node_id
            del net.op[index+1:-1]
        else:
            del net.op[index+1:]
        del net.output_info[:]
        op = net.op[index]
        index += 1

        output_tensors = []
        output_shapes = []
        op_name = op.name
        if is_quantize:
            op.name = MaceKeyword.mace_output_node_name + '_' + op.name
        if is_hexagon:
            mace_check(len(op.output) == 1,
                       "Only supports number of outputs of Hexagon op be 1.")
        for i in range(len(op.output)):
            output_tensors.append(str(op.output[i]))
            output_shapes.append(
                ",".join([str(dim) for dim in op.output_shape[i].dims]))
            # modify output info
            output_info = net.output_info.add()
            output_info.name = op.output[i]
            output_info.data_format = data_format
            output_info.dims.extend(op.output_shape[i].dims)
            output_info.data_type = mace_pb2.DT_FLOAT
            if is_quantize:
                output_info.scale = op.quantize_info[0].scale
                output_info.zero_point = op.quantize_info[0].zero_point
            # modify output op
            if is_quantize:
                output_name = op.output[i]
                new_output_name = \
                    MaceKeyword.mace_output_node_name + '_' + op.output[i]
                op.output[i] = new_output_name
                if not is_hexagon:
                    dequantize_op = net.op.add()
                    dequantize_op.name = normalize_op_name(output_name)
                    dequantize_op.type = MaceOp.Dequantize.name
                    dequantize_op.input.append(new_output_name)
                    dequantize_op.output.append(output_name)
                    output_shape = dequantize_op.output_shape.add()
                    output_shape.dims.extend(op.output_shape[i].dims)
                    dequantize_op.output_type.append(mace_pb2.DT_FLOAT)
                    ConverterUtil.add_data_type_arg(dequantize_op,
                                                    mace_pb2.DT_UINT8)
                else:
                    dequantize_op = net.op[-1]
                    dequantize_op.name = normalize_op_name(output_name)
                    del dequantize_op.input[:]
                    del dequantize_op.output[:]
                    dequantize_op.input.append(new_output_name)
                    dequantize_op.node_input[0].node_id = op.node_id
                    dequantize_op.output.append(output_name)
                    if dequantize_op.type == \
                            HexagonOp.DequantizeOUTPUT_8tof.name:
                        input_min = new_output_name[:-1] + '1'
                        input_max = new_output_name[:-1] + '2'
                        dequantize_op.input.extend([input_min, input_max])
                        dequantize_op.node_input[1].node_id = op.node_id
                        dequantize_op.node_input[2].node_id = op.node_id
                        del dequantize_op.node_input[3:]
                    else:
                        del dequantize_op.node_input[1:]

        model_path = save_model_to_proto(net, normalize_op_name(op_name),
                                         output_dir)
        output_config = {"model_file_path": str(model_path),
                         "output_tensors": output_tensors,
                         "output_shapes": output_shapes}
        output_configs["subgraphs"].append(output_config)

    output_configs_path = output_dir + "outputs.yml"
    with open(output_configs_path, "w") as f:
        yaml.dump(output_configs, f, default_flow_style=False)


def get_layers(model_dir, model_name, layers):
    model_file = "%s/%s.pb" % (model_dir, model_name)
    output_dir = "%s/output_models/" % model_dir
    if os.path.exists(output_dir):
        sh.rm('-rf', output_dir)
    os.makedirs(output_dir)

    convert(model_file, output_dir, layers)

    output_configs_path = output_dir + "outputs.yml"
    with open(output_configs_path) as f:
        output_configs = yaml.load(f)
    output_configs = output_configs['subgraphs']

    return output_configs


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="pb file to load.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save the output graph to.")
    parser.add_argument(
        "--layers",
        type=str,
        default="-1",
        help="'start_layer:end_layer' or 'layer', similar to python slice."
             " Use with --validate flag.")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, _ = parse_args()
    convert(FLAGS.model_file, FLAGS.output_dir, FLAGS.layers)
