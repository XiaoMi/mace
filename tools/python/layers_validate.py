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
from utils.config_parser import DataFormat
from utils.config_parser import ModelKeys
from utils.util import mace_check


def normalize_op_name(name):
    return name.replace('/', '_').replace(':', '_')


def save_model_to_proto(net_def, model_name, output_dir):
    output_path = output_dir + "/" + model_name + ".pb"
    with open(output_path, "wb") as f:
        f.write(net_def.SerializeToString())
    with open(output_path + "_txt", "w") as f:
        f.write(str(net_def))
    return output_path


def init_multi_net_def(model_file):
    mace_check(os.path.isfile(model_file),
               "Input graph file '" + model_file + "' does not exist!")
    multi_net_def = mace_pb2.MultiNetDef()
    with open(model_file, "rb") as f:
        multi_net_def.ParseFromString(f.read())
    return multi_net_def


class MultiNetDefInfo:
    def __init__(self, multi_net_def, layers):
        self.init_multi_net_def_info(multi_net_def)
        self.EndIndexDecrement()
        self.handle_index(layers)

    def init_multi_net_def_info(self, multi_net_def):
        netdefs = multi_net_def.net_def
        self.net_num = len(netdefs)
        self.net_defs = [None] * self.net_num
        self.net_op_nums = [0] * self.net_num
        self.quantizes = [False] * self.net_num
        self.hexagons = [False] * self.net_num
        for net_def in netdefs:
            order = net_def.infer_order
            self.net_defs[order] = net_def
            self.net_op_nums[order] = len(net_def.op)
            is_quantize = ConverterUtil.get_arg(
                net_def, MaceKeyword.mace_quantize_flag_arg_str)
            self.quantizes[order] = \
                False if is_quantize is None else is_quantize.i == 1
            self.hexagons[order] = \
                self.quantizes[order] and \
                (net_def.op[-1].type == HexagonOp.DequantizeOUTPUT_8tof.name or
                 net_def.op[-1].type == HexagonOp.OUTPUT.name)

        self.end_index = self.start_index = 0
        for op_num in self.net_op_nums:
            self.end_index = self.end_index + op_num
        self.start_net_idx = 0
        self.start_op_idx = 0
        self.end_net_idx = self.net_num
        self.end_op_idx = self.net_op_nums[self.end_net_idx - 1]

    def handle_index(self, layers):
        num_layers = self.end_index - self.start_index + 1
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
        start_index += self.start_index
        end_index += self.start_index
        start_index = \
            max(self.start_index, min(self.end_index - 1, start_index))
        end_index = max(self.start_index + 1, min(self.end_index, end_index))

        for i in range(self.net_num):
            start_index = start_index - self.net_op_nums[i]
            if start_index < 0:
                self.start_net_idx = i
                self.start_op_idx = start_index + self.net_op_nums[i]
                break
        for i in range(self.net_num):
            end_index = end_index - self.net_op_nums[i]
            if end_index < 0:
                self.end_net_idx = i
                self.end_op_idx = end_index + self.net_op_nums[i]
                break

    def StartIndexIncrement(self):
        self.start_index = self.start_index + 1
        if self.start_op_idx + 1 < self.net_op_nums[self.start_net_idx]:
            self.start_op_idx = self.start_op_idx + 1
        else:
            self.start_net_idx = self.start_net_idx + 1
            mace_check(self.start_net_idx < self.net_num, "invalid index.")
            self.start_op_idx = 0

    def EndIndexDecrement(self):
        self.end_index = self.end_index - 1
        if self.end_op_idx > 0:
            self.end_op_idx = self.end_op_idx - 1
        else:
            self.end_net_idx = self.end_net_idx - 1
            mace_check(self.end_net_idx > 0)
            self.end_op_idx = self.net_op_nums[self.end_net_idx - 1] - 1

    def index_valid(self):
        if self.start_net_idx < self.end_net_idx:
            return True
        if self.start_op_idx < self.end_op_idx:
            return True
        return False

    def get_current_op(self):
        return self.net_defs[self.start_net_idx].op[self.start_op_idx]

    def get_next_op(self):
        if self.start_op_idx + 1 < self.net_op_nums[self.start_net_idx]:
            return self.net_defs[self.start_net_idx].op[self.start_op_idx + 1]
        elif self.start_net_idx + 1 < self.net_num:
            return self.net_defs[self.start_net_idx + 1].op[0]
        else:
            return None

    def get_current_net_idx(self):
        return self.start_net_idx

    def get_current_op_idx(self):
        return self.start_op_idx

    def is_hexagon(self, net_idx):
        return self.hexagons[net_idx]

    def is_quantize(self, net_idx):
        return self.quantizes[net_idx]


def convert(model_file, output_dir, layers):
    mace_check(os.path.isdir(output_dir),
               "Output directory '" + output_dir + "' does not exist!")
    multi_net_def = init_multi_net_def(model_file)
    multi_net_info = MultiNetDefInfo(multi_net_def, layers)

    output_configs = {ModelKeys.subgraphs: []}
    while multi_net_info.index_valid():
        # omit BatchToSpaceND and op before that due to changed graph
        cur_op = multi_net_info.get_current_op()
        next_op = multi_net_info.get_next_op()
        if cur_op.type == MaceOp.BatchToSpaceND.name or \
                cur_op.type == HexagonOp.BatchToSpaceND_8.name or \
                (cur_op.type == MaceOp.Quantize.name or
                 cur_op.type == HexagonOp.QuantizeINPUT_f_to_8.name or
                 cur_op.type == HexagonOp.INPUT.name) or \
                (cur_op.type == MaceOp.Dequantize.name or
                 cur_op.type == HexagonOp.DequantizeOUTPUT_8tof.name or
                 cur_op.type == HexagonOp.OUTPUT.name) or \
                (next_op is not None and
                 (next_op.type == MaceOp.BatchToSpaceND.name or
                  next_op.type == HexagonOp.BatchToSpaceND_8.name)) or \
                cur_op.name.startswith(MaceKeyword.mace_output_node_name):
            multi_net_info.StartIndexIncrement()
            continue
        multi_net = copy.deepcopy(multi_net_def)
        net_defs = multi_net.net_def

        # remove unused net_def
        net = None
        cur_net_idx = multi_net_info.get_current_net_idx()
        for net_def in net_defs[:]:
            if net_def.infer_order == cur_net_idx:
                net = net_def
            elif net_def.infer_order > cur_net_idx:
                net_defs.remove(net_def)
        del multi_net.output_tensor[:]

        # remove unsued op
        cur_op_idx = multi_net_info.get_current_op_idx()
        is_hexagon = multi_net_info.is_hexagon(cur_net_idx)
        if is_hexagon:
            # reuse dequantize op and it's min/max tensor's node_id
            del net.op[(cur_op_idx + 1):-1]
        else:
            del net.op[(cur_op_idx + 1):]
        del net.output_info[:]
        op = net.op[cur_op_idx]
        multi_net_info.StartIndexIncrement()

        output_tensors = []
        output_shapes = []
        output_data_types = []
        output_data_formats = []
        op_name = op.name
        if str(op.name).startswith(MaceKeyword.mace_output_node_name):
            continue
        is_quantize = multi_net_info.is_quantize(cur_net_idx)
        if is_quantize:
            op.name = MaceKeyword.mace_output_node_name + '_' + op.name
        if is_hexagon:
            if len(op.output) != 1:
                print("Skip %s(%s)" % (op.name, op.type))
                continue
        data_format = ConverterUtil.get_arg(
                op, MaceKeyword.mace_data_format_str)
        for i in range(len(op.output)):
            output_tensors.append(str(op.output[i]))
            output_shapes.append(
                ",".join([str(dim) for dim in op.output_shape[i].dims]))
            if data_format.i == DataFormat.NONE.value:
                output_data_formats.append(DataFormat.NONE.name)
            elif data_format.i == DataFormat.NCHW.value:
                output_data_formats.append(DataFormat.NCHW.name)
            else:
                output_data_formats.append(DataFormat.NHWC.name)
            # modify output info
            multi_net.output_tensor.append(op.output[i])
            output_info = net.output_info.add()
            output_info.name = op.output[i]
            output_info.data_format = data_format.i
            output_info.dims.extend(op.output_shape[i].dims)
            data_type = ConverterUtil.get_arg(
                            op, MaceKeyword.mace_op_data_type_str)
            if mace_pb2.DT_INT32 == data_type.i:
                output_info.data_type = mace_pb2.DT_INT32
                output_data_types.append('int32')
            else:
                output_info.data_type = mace_pb2.DT_FLOAT
                output_data_types.append('float32')
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

        model_path = save_model_to_proto(multi_net, normalize_op_name(op_name),
                                         output_dir)
        output_config = {ModelKeys.model_file_path: str(model_path),
                         ModelKeys.output_tensors: output_tensors,
                         ModelKeys.output_shapes: output_shapes,
                         "output_data_formats": output_data_formats,
                         ModelKeys.output_data_types: output_data_types}
        output_configs[ModelKeys.subgraphs].append(output_config)

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
