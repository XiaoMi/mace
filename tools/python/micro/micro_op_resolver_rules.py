from enum import Enum
from py_proto import mace_pb2
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from utils.config_parser import DataFormat
from utils.config_parser import ModelKeys
from utils.config_parser import Platform
from utils.util import mace_check
from utils.net_util import NetUtil


def scratch_zero_size(mace_op, mace_net):
    return 0


def scratch_pooling(mace_op, mace_net):
    input_dims = NetUtil.get_input_dims(mace_op, mace_net, 0)
    channels = input_dims[3]
    return channels * (4 + 4)


class MicroOPSResolverRule:

    def __init__(self, header_path, class_name, mace_op_type, data_type,
                 priority, tag=None, scratch_fun=scratch_zero_size):
        self._header_path = header_path
        self._class_name = class_name
        self._mace_op_type = mace_op_type
        self._data_type = data_type
        self._priority = priority
        self._tag = tag
        self._scratch_fun = scratch_fun

    def class_name(self, mace_op, mace_net):
        return self._class_name

    def header_path(self, mace_op, mace_net):
        return self._header_path

    def priority(self, mace_op, mace_net):
        return self._priority

    def scratch(self, mace_op, mace_net):
        return self._scratch_fun(mace_op, mace_net)

    def valid_data_type(self, mace_op, mace_net):
        arg = NetUtil.get_arg(mace_op, MaceKeyword.mace_op_data_type_str)
        mace_check(arg is not None, "mace_op should has a explicit data type")
        if (self._data_type == mace_pb2.DT_FLOAT):
            return arg.i == mace_pb2.DT_FLOAT or arg.i == mace_pb2.DT_BFLOAT16
        else:
            return arg.i == self._data_type

    def valid_op(self, mace_op, mace_net):
        return mace_op.type == self._mace_op_type

    def valid_tag(self, mace_op, mace_net):
        return True

    def valid(self, mace_op, mace_net):
        return self.valid_op(mace_op, mace_net) and \
            self.valid_data_type(mace_op, mace_net) and \
            self.valid_tag(mace_op, mace_net)


class MicroConvOptOPSResolverRule(MicroOPSResolverRule):

    def valid_tag(self, mace_op, mace_net):
        tag = ""
        output_shape = mace_op.output_shape[0].dims
        size = output_shape[0] * output_shape[1] * output_shape[2]
        if size >= 4:
            size = 4
        channel = output_shape[3]
        if channel >= 4:
            channel = 4
        if channel >= 2 and size >= 4:
            tag = "c%ss%s" % (channel, size)

        return tag == self._tag


class MicroDepthwiseConvOptOPSResolverRule(MicroOPSResolverRule):

    def valid_tag(self, mace_op, mace_net):
        tag = ""
        output_shape = mace_op.output_shape[0].dims
        size = output_shape[0] * output_shape[1] * output_shape[2]
        if size >= 4:
            size = 4
        filter_dims = NetUtil.get_input_dims(mace_op, mace_net, 1)
        k_batch = filter_dims[0]
        if k_batch >= 4:
            k_batch = 4
        if size >= 4:
            tag = "kb%ss%s" % (k_batch, size)

        return tag == self._tag


class MicroPoolingOptOPSResolverRule(MicroOPSResolverRule):

    def valid_tag(self, mace_op, mace_net):
        tag = ""
        kernels = NetUtil.get_arg(mace_op, MaceKeyword.mace_kernel_str)
        mace_check(kernels is not None, "Get kernels failed.")
        size = kernels.ints[0] * kernels.ints[1]
        if size >= 4:
            tag = "s4"

        return tag == self._tag


RefOPSResolverRules = [
    MicroOPSResolverRule(
        "micro/ops/argmax.h",
        "ArgMaxOp<mifloat>",
        MaceOp.ArgMax.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        "micro/ops/nhwc/conv_2d_ref.h",
        "Conv2dRefOp",
        MaceOp.Conv2D.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/cast.h',
        'CastOp',
        MaceOp.Cast.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/nhwc/pooling_ref.h',
        'PoolingRefOp',
        MaceOp.Pooling.name,
        mace_pb2.DT_FLOAT,
        1,
        scratch_fun=scratch_pooling
    ),
    MicroOPSResolverRule(
        'micro/ops/squeeze.h',
        'SqueezeOp',
        MaceOp.Squeeze.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/softmax.h',
        'SoftmaxOp',
        MaceOp.Softmax.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/eltwise.h', 'EltwiseOp<mifloat>',
        MaceOp.Eltwise.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/activation.h', 'ActivationOp',
        MaceOp.Activation.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/strided_slice.h', 'StridedSliceOp<mifloat>',
        MaceOp.StridedSlice.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/reduce.h', 'ReduceOp<mifloat>', MaceOp.Reduce.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/stack.h', 'StackOp<mifloat>', MaceOp.Stack.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/bias_add.h', 'BiasAddOp', MaceOp.BiasAdd.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/nhwc/batch_norm.h', 'BatchNormOp',
        MaceOp.BatchNorm.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/matmul.h', 'MatMulOp', MaceOp.MatMul.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/shape.h', 'ShapeOp', MaceOp.Shape.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/reshape.h', 'ReshapeOp<mifloat>',
        MaceOp.Reshape.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/expand_dims.h', 'ExpandDimsOp',
        MaceOp.ExpandDims.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/concat.h', 'ConcatOp<mifloat>', MaceOp.Concat.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    MicroOPSResolverRule(
        'micro/ops/nhwc/depthwise_conv_2d_ref.h',
        'DepthwiseConv2dRefOp',
        MaceOp.DepthwiseConv2d.name,
        mace_pb2.DT_FLOAT,
        1
    ),
    # INT8
    MicroOPSResolverRule(
        'micro/ops/reshape.h', 'ReshapeOp<int8_t>',
        MaceOp.Reshape.name,
        mace_pb2.DT_INT8,
        1
    )
]


OptOPSResolverRules = [
    MicroConvOptOPSResolverRule(
        'micro/ops/nhwc/conv_2d_c4_s4.h',
        'Conv2dC4S4Op',
        MaceOp.Conv2D.name,
        mace_pb2.DT_FLOAT,
        10,
        'c4s4'
    ),
    MicroConvOptOPSResolverRule(
        'micro/ops/nhwc/conv_2d_c3_s4.h',
        'Conv2dC3S4Op',
        MaceOp.Conv2D.name,
        mace_pb2.DT_FLOAT,
        10,
        'c3s4'
    ),
    MicroConvOptOPSResolverRule(
        'micro/ops/nhwc/conv_2d_c2_s4.h',
        'Conv2dC2S4Op',
        MaceOp.Conv2D.name,
        mace_pb2.DT_FLOAT,
        10,
        'c2s4'
    ),
    MicroDepthwiseConvOptOPSResolverRule(
        'micro/ops/nhwc/depthwise_conv_2d_kb4_s4.h',
        'DepthwiseConv2dKB4S4Op',
        MaceOp.DepthwiseConv2d.name,
        mace_pb2.DT_FLOAT,
        10,
        'kb4s4'
    ),
    MicroDepthwiseConvOptOPSResolverRule(
        'micro/ops/nhwc/depthwise_conv_2d_kb3_s4.h',
        'DepthwiseConv2dKB3S4Op',
        MaceOp.DepthwiseConv2d.name,
        mace_pb2.DT_FLOAT,
        10,
        'kb3s4'
    ),
    MicroDepthwiseConvOptOPSResolverRule(
        'micro/ops/nhwc/depthwise_conv_2d_kb2_s4.h',
        'DepthwiseConv2dKB2S4Op',
        MaceOp.DepthwiseConv2d.name,
        mace_pb2.DT_FLOAT,
        10,
        'kb2s4'
    ),
    MicroPoolingOptOPSResolverRule(
        'micro/ops/nhwc/pooling_s4.h', 'PoolingS4Op',
        MaceOp.Pooling.name,
        mace_pb2.DT_FLOAT,
        10,
        "s4",
        scratch_fun=scratch_pooling
    )
]
