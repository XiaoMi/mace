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


def scratch_cmsis_conv_2d(mace_op, mace_net):
    output_channels = mace_op.output_shape[0].dims[3]
    bias_bytes = output_channels * 4
    cmsis_quant_bytes = output_channels * 4 * 2
    input_dims = NetUtil.get_input_dims(mace_op, mace_net, 0)
    filter_dims = NetUtil.get_input_dims(mace_op, mace_net, 1)
    cmsis_nn_buffer_bytes = input_dims[3] * filter_dims[2] * filter_dims[1] * 4
    return cmsis_nn_buffer_bytes + bias_bytes + cmsis_quant_bytes


def scratch_cmsis_depthwise_conv_2d(mace_op, mace_net):
    output_channels = mace_op.output_shape[0].dims[3]
    bias_bytes = output_channels * 4
    cmsis_quant_bytes = output_channels * 2
    input_dims = NetUtil.get_input_dims(mace_op, mace_net, 0)
    filter_dims = NetUtil.get_input_dims(mace_op, mace_net, 1)
    cmsis_nn_buffer_bytes = input_dims[3] * filter_dims[2] * filter_dims[1] * 2
    return cmsis_nn_buffer_bytes + bias_bytes + cmsis_quant_bytes


def scratch_cmsis_matmul(mace_op, mace_net):
    output_channels = mace_op.output_shape[0].dims[1]
    bias_bytes = output_channels * 4
    return bias_bytes


def scratch_cmsis_pooling(mace_op, mace_net):
    input_dims = NetUtil.get_input_dims(mace_op, mace_net, 0)
    channels = input_dims[3]
    return channels * 2


CmsisOPSResolverRules = [
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/arm_eltwise_int8.h',
        'ArmEltwiseInt8Op',
        MaceOp.Eltwise.name,
        mace_pb2.DT_INT8,
        100
    ),
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/quantize.h',
        'QuantizeOp',
        MaceOp.Quantize.name, mace_pb2.DT_INT8,
        100
    ),
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/dequantize.h',
        'DequantizeOp',
        MaceOp.Dequantize.name, mace_pb2.DT_INT8,
        100
    ),
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/arm_conv_2d_int8.h',
        'ArmConv2dInt8Op',
        MaceOp.Conv2D.name, mace_pb2.DT_INT8,
        100,
        scratch_fun=scratch_cmsis_conv_2d
    ),
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/arm_depthwise_conv_2d_int8.h',
        'ArmDepthwiseConv2dInt8Op',
        MaceOp.DepthwiseConv2d.name, mace_pb2.DT_INT8,
        100,
        scratch_fun=scratch_cmsis_depthwise_conv_2d
    ),
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/arm_pooling_int8.h',
        'ArmPoolingInt8Op',
        MaceOp.Pooling.name, mace_pb2.DT_INT8,
        100,
        scratch_fun=scratch_cmsis_pooling
    ),
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/arm_softmax_int8.h',
        'ArmSoftmaxInt8Op',
        MaceOp.Softmax.name, mace_pb2.DT_INT8,
        100
    ),
    MicroOPSResolverRule(
        'micro/ops/cmsis_nn/arm_mat_mul_int8.h',
        'ArmMatMulInt8Op',
        MaceOp.MatMul.name, mace_pb2.DT_INT8,
        100,
        scratch_fun=scratch_cmsis_matmul
    ),
]
