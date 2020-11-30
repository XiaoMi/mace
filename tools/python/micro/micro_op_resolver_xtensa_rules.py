from enum import Enum
from micro.micro_op_resolver_rules import MicroOPSResolverRule
from py_proto import mace_pb2
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from utils.config_parser import DataFormat
from utils.config_parser import ModelKeys
from utils.config_parser import Platform
from utils.util import mace_check
from utils.net_util import NetUtil


ALIGNMENT = 8
BUS_WIDTH = 8
OUT_HEIGHT_PER_ITER = 2


def aligned_size(size, align):
    return (size + align - 1) & ~(align - 1)


def scratch_xtensa_matmul(mace_op, mace_net):
    output_channels = mace_op.output_shape[0].dims[1]
    bias_bytes = output_channels * 4
    return bias_bytes


def scratch_xtensa_conv_2d(mace_op, mace_net):
    output_channels = mace_op.output_shape[0].dims[3]
    bias_bytes = output_channels * 4

    input_dims = NetUtil.get_input_dims(mace_op, mace_net, 0)
    input_height = input_dims[1]
    input_width = input_dims[2]
    input_channels = input_dims[3]

    output_dims = mace_op.output_shape[0].dims
    out_height = output_dims[1]
    out_width = output_dims[2]

    filter_dims = NetUtil.get_input_dims(mace_op, mace_net, 1)
    kernel_height = filter_dims[1]
    kernel_width = filter_dims[2]

    strides = NetUtil.get_arg(mace_op, "strides").ints
    x_stride = strides[0]
    y_stride = strides[1]

    padding = NetUtil.calc_padding(mace_op, mace_net)
    x_padding = padding[0]
    y_padding = padding[1]

    # xa_nn_conv2d_std_getsize
    mem_req = 0
    input_size = 0
    align_size = 0

    mem_req += 12 + ALIGNMENT - 1
    data_type = NetUtil.get_arg(mace_op, "T").i
    if data_type == mace_pb2.DT_FLOAT:
        input_size = 4
        align_size = ALIGNMENT >> 2
    else:
        mace_check(False, "Unsupported")

    y_b_pad = kernel_height + (out_height - 1) * \
        y_stride - (y_padding + input_height)
    y_b_pad = max(0, y_b_pad)
    input_channels_pad = aligned_size(input_channels, align_size)
    cir_buf_size_bytes = (y_padding + input_height + y_b_pad) * \
        kernel_width * input_channels_pad * input_size

    mem_req += cir_buf_size_bytes
    mem_req += BUS_WIDTH

    return int(mem_req * 4 + bias_bytes)


def scratch_xtensa_depthwise_conv_2d(mace_op, mace_net):
    output_channels = mace_op.output_shape[0].dims[3]
    bias_bytes = output_channels * 4

    input_dims = NetUtil.get_input_dims(mace_op, mace_net, 0)
    input_height = input_dims[1]
    input_width = input_dims[2]
    input_channels = input_dims[3]

    output_dims = mace_op.output_shape[0].dims
    output_height = output_dims[1]
    output_width = output_dims[2]

    filter_dims = NetUtil.get_input_dims(mace_op, mace_net, 1)
    kernel_height = filter_dims[1]
    kernel_width = filter_dims[2]
    channels_multiplier = filter_dims[0]

    strides = NetUtil.get_arg(mace_op, "strides").ints
    x_stride = strides[0]
    y_stride = strides[1]

    padding = NetUtil.calc_padding(mace_op, mace_net)
    x_padding = padding[0]
    y_padding = padding[1]

    # xa_nn_conv2d_depthwise_getsize
    data_type = NetUtil.get_arg(mace_op, "T").i
    # data_format = NetUtil.get_arg(mace_op, "data_format").i
    if data_type == mace_pb2.DT_FLOAT:
        scratch_bytewidth = 4  # f32 scratch
        circ_buf_bytewidth = 4  # bytewidth
        bytewidth = circ_buf_bytewidth
    else:
        mace_check(False, "Unsupported")

    state_size = aligned_size(24, ALIGNMENT)

    circ_buf_height = kernel_height + ((output_height - 1) * y_stride)
    circ_buf_height = max(circ_buf_height, y_padding + input_height)

    if bytewidth == 4:
        circ_buf_channels = aligned_size(input_channels*channels_multiplier, 2)
    else:
        circ_buf_channels = aligned_size(input_channels*channels_multiplier, 4)

    size_in_bytes = bytewidth*circ_buf_height*circ_buf_channels*kernel_width
    circ_buf_size = size_in_bytes

    xtensa_total_size = state_size + circ_buf_size

    return xtensa_total_size * 4 + bias_bytes


XtensaOPSResolverRules = [
    MicroOPSResolverRule(
        "micro/ops/xtensa/matmul_xtensa.h",
        "MatMulXtensaOp",
        MaceOp.MatMul.name, mace_pb2.DT_FLOAT,
        100,
        scratch_fun=scratch_xtensa_matmul
    ),
    MicroOPSResolverRule(
        "micro/ops/xtensa/conv_2d_xtensa.h",
        "Conv2dXtensaOp",
        MaceOp.Conv2D.name, mace_pb2.DT_FLOAT,
        100,
        scratch_fun=scratch_xtensa_conv_2d
    ),
    MicroOPSResolverRule(
        "micro/ops/xtensa/depthwise_conv_2d_xtensa.h",
        "DepthwiseConv2dXtensaOp",
        MaceOp.DepthwiseConv2d.name, mace_pb2.DT_FLOAT,
        100,
        scratch_fun=scratch_xtensa_depthwise_conv_2d
    )
]
