from py_proto import mace_pb2

from utils.config_parser import DataFormat
from utils.config_parser import DeviceType
from utils.config_parser import Platform
from utils.util import mace_check
from transform.base_converter import PaddingMode


class NetUtil(object):

    @staticmethod
    def get_arg(op, arg_name):
        for arg in op.arg:
            if arg.name == arg_name:
                return arg
        mace_check(False, "%s arg is not exist" % arg_name)

    @staticmethod
    def get_input_dims(mace_op, mace_net, idx):
        input_name = mace_op.input[idx]
        for const_tensor in mace_net.tensors:
            if input_name == const_tensor.name:
                return const_tensor.dims
        for pre_op in mace_net.op:
            for i in range(len(pre_op.output)):
                if input_name == pre_op.output[i]:
                    return pre_op.output_shape[i].dims
        for input_info in mace_net.input_info:
            if input_name == input_info.name:
                return input_info.dims
        mace_check(False, "unreachable")

    @staticmethod
    def calc_padding(mace_op, mace_net):
        input_dims = NetUtil.get_input_dims(mace_op, mace_net, 0)
        input_height = input_dims[1]
        input_width = input_dims[2]

        filter_dims = NetUtil.get_input_dims(mace_op, mace_net, 1)
        kernel_height = filter_dims[1]
        kernel_width = filter_dims[2]

        dilations = NetUtil.get_arg(mace_op, "dilations").ints
        strides = NetUtil.get_arg(mace_op, "strides").ints

        k_extent_height = (kernel_height - 1) * dilations[0] + 1
        k_extent_width = (kernel_width - 1) * dilations[1] + 1

        padding_type = NetUtil.get_arg(mace_op, "padding").i

        if padding_type == PaddingMode.VALID.value:
            output_height = \
                int((input_height - k_extent_height) / strides[0]) + 1
            output_width = int((input_width - k_extent_width) / strides[1]) + 1
        elif padding_type == PaddingMode.SAME.value:
            output_height = int((input_height - 1) / strides[0]) + 1
            output_width = int((input_width - 1) / strides[1]) + 1
        elif padding_type == PaddingMode.FULL.value:
            output_height = \
                int((input_height + k_extent_height - 2) / strides[0]) + 1
            output_width = \
                int((input_width + k_extent_width - 2) / strides[1]) + 1
        else:
            mace_check(False, "Unsupported padding type: %d" % padding_type)

        padding0 = max(
            0,
            (output_height - 1) * strides[0] + k_extent_height - input_height)
        padding1 = max(
            0,
            (output_width - 1) * strides[1] + k_extent_width - input_width)

        padding0 = int(padding0 / 2)
        padding1 = int(padding1 / 2)

        return [padding0, padding1]
