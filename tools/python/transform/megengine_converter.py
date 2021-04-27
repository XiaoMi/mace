import os
import math
import numpy as np
import six
import megengine._internal as mgb
from enum import Enum

from py_proto import mace_pb2
from transform import base_converter
from transform.base_converter import PoolingType
from transform.base_converter import ActivationType
from transform.base_converter import EltwiseType
from transform.base_converter import FrameworkType
from transform.base_converter import ReduceType
from transform.base_converter import DataFormat
from transform.base_converter import MaceOp
from transform.base_converter import MaceKeyword
from transform.base_converter import ConverterUtil
from transform.base_converter import RoundMode
from utils.util import mace_check

mge_kernel_h_str = "window_h"
mge_kernel_w_str = "window_w"
mge_stride_h_str = "stride_h"
mge_stride_w_str = "stride_w"
mge_pad_h_str = "pad_h"
mge_pad_w_str = "pad_w"
mge_dilate_h_str = "dilate_h"
mge_dilate_w_str = "dilate_w"

MGESupportedOps = [
    "AxisAddRemove",
    "BatchNormForward",
    "Concat",
    "ConvolutionForward",
    "ConvolutionBackwardData",
    "Dimshuffle",
    "Elemwise",
    "GetVarShape",
    "Host2DeviceCopy",
    "Identity",
    "MarkNoBroadcastElemwise",
    "MatrixMul",
    "PoolingForward",
    "Reduce",
    "Reshape",
    "SharedDeviceTensor",
    "Subtensor",
]

MGEOpType = Enum("MGEOpType", [(op, op) for op in MGESupportedOps], type=str)


def get_symvar_value(mge_symvar):
    if mge_symvar.inferred_value is not None:
        val = mge_symvar.inferred_value
    else:
        cg = mge_symvar.owner_graph
        func = cg.compile_outonly(mge_symvar)
        val = func()

    return val


def is_consumer_group_conv(mge_symvar, var2oprs, map_oprs):
    consumer_ids = var2oprs[mge_symvar.id]
    n_consumers = len(consumer_ids)
    for consumer_id in consumer_ids:
        consumer_op = map_oprs[consumer_id[0]]
        if (mgb.cgtools.get_opr_type(consumer_op)
                in ("ConvolutionForward", "ConvolutionBackwardData")
                and consumer_op.params["sparse"] == "GROUP"):
            mace_check(n_consumers == 1,
                       "This tensor should only feed depthwise conv/deconv")
            return True
    return False


class MegengineConverter(base_converter.ConverterInterface):
    """A class for convert megengine dumped model to mace model."""

    compute_format_type = {
        "NCHW": DataFormat.NCHW,
        "NHWC": DataFormat.NHWC,
        "DEFAULT": DataFormat.NCHW,
    }

    reduce_math_type = {
        "SUM": ReduceType.SUM,
        "PROD": ReduceType.PROD,
        "MIN": ReduceType.MIN,
        "MAX": ReduceType.MAX,
    }

    # SQE_DIFF, CLIP, SIGN maybe needed
    eltwise_type = {
        "ADD": EltwiseType.SUM,
        "SUB": EltwiseType.SUB,
        "MUL": EltwiseType.PROD,
        "TRUE_DIV": EltwiseType.DIV,
        "MIN": EltwiseType.MIN,
        "MAX": EltwiseType.MAX,
        "NEGATE": EltwiseType.NEG,
        "ABS": EltwiseType.ABS,
        "POW": EltwiseType.POW,
        "EQ": EltwiseType.EQUAL,
        "FLOOR_DIV": EltwiseType.FLOOR_DIV,
        "EXP": EltwiseType.POW,
    }

    activation_type = {
        "RELU": ActivationType.RELU,
        "TANH": ActivationType.TANH,
        "SIGMOID": ActivationType.SIGMOID,
    }

    def __init__(self, option, src_model_file):
        self._op_converters = {
            MGEOpType.AxisAddRemove.name: self.convert_axisaddrm,
            MGEOpType.BatchNormForward.name: self.convert_batchnorm,
            MGEOpType.Concat.name: self.convert_concat,
            MGEOpType.ConvolutionForward.name: self.convert_conv2d,
            MGEOpType.ConvolutionBackwardData.name: self.convert_deconv2d,
            MGEOpType.Dimshuffle.name: self.convert_dimshuffle,
            MGEOpType.Elemwise.name: self.convert_elemwise,
            MGEOpType.GetVarShape.name: self.convert_shape,
            MGEOpType.Host2DeviceCopy.name: self.convert_nop,
            MGEOpType.Identity.name: self.convert_identity,
            MGEOpType.MarkNoBroadcastElemwise.name: self.convert_identity,
            MGEOpType.MatrixMul.name: self.convert_matmul,
            MGEOpType.PoolingForward.name: self.convert_pooling,
            MGEOpType.Reduce.name: self.convert_reduce,
            MGEOpType.Reshape.name: self.convert_reshape,
            MGEOpType.SharedDeviceTensor.name: self.convert_nop,
            MGEOpType.Subtensor.name: self.convert_subtensor,
        }
        self._option = option
        self._converter_info = dict()
        self._mace_net_def = mace_pb2.NetDef()
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.OIHW)
        ConverterUtil.add_data_format_arg(self._mace_net_def, DataFormat.NCHW)

        cg, _, outputs = mgb.load_comp_graph_from_file(src_model_file)
        map_oprs, _, var2oprs, *_ = mgb.cgtools.graph_traversal(outputs)
        # prune second input of reshape
        # because it introduces several ops, may increase the overhead
        operators = mgb.cgtools.get_oprs_seq(outputs, prune_reshape=True)

        self._mge_cg = cg
        self._mge_operators = operators
        self._mge_map_oprs = map_oprs
        self._mge_var2oprs = var2oprs

        self._skip_tensors = set()
        self._bn_statistis_tensors = {}

    def run(self):
        self.convert_ops()

        self.replace_input_output_tensor_name()
        return self._mace_net_def, self._converter_info

    # only change the input/output tensor name for whole model
    def replace_input_output_tensor_name(self):
        for op in self._mace_net_def.op:
            for i in six.moves.range(len(op.input)):
                if "," in op.input[i]:
                    op_name = op.input[i]
                    op_name = op_name.replace(",", "#")
                    if (op_name in self._option.input_nodes or
                            op_name in self._option.output_nodes):
                        op.input[i] = op_name
            for i in six.moves.range(len(op.output)):
                if "," in op.output[i]:
                    op_name = op.output[i]
                    op_name = op_name.replace(",", "#")
                    if op_name in self._option.output_nodes:
                        op.output[i] = op_name

    # this method will be called by convert_conv2d/deconv2d and convert_pooling
    @staticmethod
    def add_stride_pad_kernel_arg(params, op_def):
        stride = [params[mge_stride_h_str], params[mge_stride_w_str]]
        pad = [params[mge_pad_h_str] * 2, params[mge_pad_w_str] * 2]

        strides_arg = op_def.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(stride)
        padding_arg = op_def.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_values_str
        padding_arg.ints.extend(pad)

        if op_def.type == MaceOp.Pooling.name:
            kernel = [params[mge_kernel_h_str], params[mge_kernel_w_str]]
            kernels_arg = op_def.arg.add()
            kernels_arg.name = MaceKeyword.mace_kernel_str
            kernels_arg.ints.extend(kernel)
        if op_def.type in (MaceOp.Conv2D.name, MaceOp.DepthwiseConv2d.name,
                           MaceOp.Deconv2D.name,
                           MaceOp.DepthwiseDeconv2d.name):
            dilation = [params[mge_dilate_h_str], params[mge_dilate_w_str]]
            dilation_arg = op_def.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            dilation_arg.ints.extend(dilation)

    def convert_ops(self):
        for mge_op in self._mge_operators:
            opr_type = mgb.cgtools.get_opr_type(mge_op)

            # some reshape operators provide data for batchnorm
            if opr_type == "Reshape":
                output = mge_op.outputs[0]
                next_ops = self._mge_var2oprs[output.id]
                if len(next_ops) == 1:
                    (next_op_id, _) = next_ops[0]
                    next_op = self._mge_map_oprs[next_op_id]

                    if mgb.cgtools.get_opr_type(next_op) == "BatchNormForward":
                        self._skip_tensors.update(
                            [inp.name for inp in mge_op.inputs])
                        # using output name to address input symbol var
                        self._bn_statistis_tensors[mge_op.outputs[0].name] = \
                            mge_op.inputs[0]
                        # skip this reshape op
                        continue

            self._op_converters[opr_type](mge_op)

        self.convert_tensors()

    def add_tensor(self, name, shape, data_type, value):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        if data_type == mace_pb2.DT_INT32:
            tensor.int32_data.extend(value)
        else:
            tensor.float_data.extend(value)

    # convert all pre-calculated and constant tensors
    def convert_tensors(self):
        for mge_op in self._mge_operators:
            type_opr = mgb.cgtools.get_opr_type(mge_op)

            # all tensors generated by SharedDeviceTensor op
            if type_opr == "SharedDeviceTensor":
                output = mge_op.outputs[0]
                if output.name not in self._skip_tensors:
                    nshape = output.imm_shape
                    # tensor used for depthwise conv/deconv should be reshaped
                    for_group_conv = is_consumer_group_conv(
                        output, self._mge_var2oprs, self._mge_map_oprs
                    )
                    if for_group_conv:
                        nshape = (
                            1,
                            output.imm_shape[0],
                            output.imm_shape[3],
                            output.imm_shape[4],
                        )

                    self.add_tensor(
                        output.name,
                        nshape,
                        mace_pb2.DT_FLOAT,
                        get_symvar_value(output).flatten())
            else:
                # handle all constant values
                for const_tensor in mge_op.inputs:
                    if (const_tensor.inferred_value is not None
                            and const_tensor.name not in self._skip_tensors):
                        self.add_tensor(
                            const_tensor.name,
                            const_tensor.imm_shape,
                            mace_pb2.DT_INT32,
                            const_tensor.inferred_value.flatten())

    def convert_nop(self, mge_op):
        pass

    def convert_general_op(self, mge_op):
        op = self._mace_net_def.op.add()
        op.name = mge_op.name
        op.type = mgb.cgtools.get_opr_type(mge_op)
        op.input.extend([mge_input.name for mge_input in mge_op.inputs])
        op.output.extend([mge_output.name for mge_output in mge_op.outputs])
        for mge_output in mge_op.outputs:
            output_shape = op.output_shape.add()
            output_shape.dims.extend(mge_output.imm_shape)

        data_type_arg = op.arg.add()
        data_type_arg.name = "T"
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.MEGENGINE.value

        # check compute format of megengine
        compute_format = DataFormat.NCHW
        try:
            if "format" in mge_op.params.keys():
                compute_format = self.compute_format_type[
                    mge_op.params["format"]
                ]
        except AttributeError:
            compute_format = DataFormat.NCHW
        ConverterUtil.add_data_format_arg(op, compute_format)

        return op

    def convert_identity(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Identity.name

    def convert_conv2d(self, mge_op):
        op = self.convert_general_op(mge_op)

        if mge_op.params["sparse"] == "GROUP":
            # weight shape in group conv2d:
            # (groups, out_channel//groups, in_channels//groups, *kernel_size)
            groups_divisible = mge_op.inputs[1].imm_shape[2]
            mace_check(
                groups_divisible == 1,
                "Mace does not support group convolution yet",
            )
            op.type = MaceOp.DepthwiseConv2d.name
        elif mge_op.params["sparse"] == "DENSE":
            op.type = MaceOp.Conv2D.name
        else:
            raise Exception("Unknown sparse mode")

        mace_check(
            mge_op.params["mode"] != "CONVOLUTION",
            "Mace does not support CONVOLUTION computation mode yet",
        )

        self.add_stride_pad_kernel_arg(mge_op.params, op)

        del op.output[1:]
        del op.output_shape[1:]

    def convert_deconv2d(self, mge_op):
        op = self.convert_general_op(mge_op)

        if mge_op.params["sparse"] == "GROUP":
            # weight shape in group conv2d:
            # (groups, out_channel//groups, in_channels//groups, *kernel_size)
            groups_divisible = mge_op.inputs[0].imm_shape[2]
            mace_check(
                groups_divisible == 1,
                "Mace does not support group deconvolution yet",
            )
            op.type = MaceOp.DepthwiseConv2d.name
        elif mge_op.params["sparse"] == "DENSE":
            op.type = MaceOp.Deconv2D.name
        else:
            mace_check(False, "Unknown sparse mode")

        mace_check(
            mge_op.params["mode"] != "CONVOLUTION",
            "Mace does not support CONVOLUTION computation mode yet",
        )

        self.add_stride_pad_kernel_arg(mge_op.params, op)

        # inputs order is strange in megengine, fix it
        swaped_list = [op.input[1], op.input[0]]
        del op.input[:]
        op.input.extend(swaped_list)

        del op.output[1:]
        del op.output_shape[1:]

    def convert_dimshuffle(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Transpose.name

        dims_arg = op.arg.add()
        dims_arg.name = MaceKeyword.mace_dims_str
        dims_arg.ints.extend(mge_op.params["pattern"])

    def convert_math_elemwise(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Eltwise.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[mge_op.params["mode"]].value
        # EXP in megengine always use the np.e as base
        if mge_op.params["mode"] == "EXP":
            exp_tensor_name = mge_op.name + "_exp_base"
            exp_shape = mge_op.outputs[0].imm_shape
            exp_value = (np.e * np.ones(exp_shape)).flatten()
            self.add_tensor(
                exp_tensor_name, exp_shape, mace_pb2.DT_FLOAT, exp_value
            )
            del op.input[0]
            op.input.extend([exp_tensor_name, mge_op.inputs[0].name])

    def convert_activation(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Activation.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type[mge_op.params["mode"]].name)

    def convert_elemwise(self, mge_op):
        mode = mge_op.params["mode"]
        if mode in self.eltwise_type:
            self.convert_math_elemwise(mge_op)
        else:
            self.convert_activation(mge_op)

    def convert_pooling(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Pooling.name

        pool_type_arg = op.arg.add()
        pool_type_arg.name = MaceKeyword.mace_pooling_type_str

        round_mode_arg = op.arg.add()
        round_mode_arg.name = MaceKeyword.mace_round_mode_str
        round_mode_arg.i = RoundMode.FLOOR.value

        # check the case of counting include padding
        mode = mge_op.params["mode"]
        if mode == "AVERAGE_COUNT_EXCLUDE_PADDING" or \
                (mode == "AVERAGE" and mge_op.params["pad_w"] == 0 and
                 mge_op.params["pad_h"] == 0):
            pool_type_arg.i = PoolingType.AVG.value
        elif mode == "MAX":
            pool_type_arg.i = PoolingType.MAX.value
        else:
            mace_check(False,
                       "AVERAGE pooling should not count padding values")

        self.add_stride_pad_kernel_arg(mge_op.params, op)

        # delete workspace output, it's useless
        del op.output[1:]
        del op.output_shape[1:]

    def convert_matmul(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.MatMul.name

        transpose_a = mge_op.params["transposeA"]
        transpose_a_arg = op.arg.add()
        transpose_a_arg.name = MaceKeyword.mace_transpose_a_str
        transpose_a_arg.i = int(transpose_a)

        transpose_b = mge_op.params["transposeB"]
        transpose_b_arg = op.arg.add()
        transpose_b_arg.name = MaceKeyword.mace_transpose_b_str
        transpose_b_arg.i = int(transpose_b)

        del op.output[1:]
        del op.output_shape[1:]

    def convert_reshape(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Reshape.name

        # just use the output shape
        del op.input[1]
        t_shape = list(mge_op.outputs[0].imm_shape)
        shape_tensor_name = mge_op.name + "_dest_shape"
        self.add_tensor(
            shape_tensor_name, [len(t_shape)], mace_pb2.DT_INT32, t_shape
        )
        op.input.extend([shape_tensor_name])

    # usually after reduce operator, remove dimension with value 1
    # it's hard to just follow this operator
    # sometimes axis-add and axis-remove may exist at the same time
    # for complicated use-case, using reshape is easier
    def convert_axisaddrm(self, mge_op):
        op = self.convert_general_op(mge_op)
        if mge_op.params["nr_desc"] == 1:
            if mge_op.params["desc"][0]["method"] == 0:
                op.type = MaceOp.ExpandDims.name
            else:
                op.type = MaceOp.Squeeze.name

            axis_arg = op.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            axis_arg.i = mge_op.params["desc"][0]["axisnum"]
        else:
            op.type = MaceOp.Reshape.name

            dest_shape_tensor_name = op.name + "_dest_shape"
            dest_shape = mge_op.outputs[0].imm_shape
            self.add_tensor(
                dest_shape_tensor_name,
                (len(dest_shape),),
                mace_pb2.DT_INT32,
                dest_shape,
            )
            op.input.extend([dest_shape_tensor_name])

    def convert_reduce(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Reduce.name

        reduce_type_arg = op.arg.add()
        reduce_type_arg.name = MaceKeyword.mace_reduce_type_str
        reduce_type_arg.i = self.reduce_math_type[mge_op.params["mode"]].value

        # in megengine axis won't be list, just int
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.ints.append(mge_op.params["axis"])

        # megengine will always keep dims in Reduce operator
        # dim removal will be done by operator AxisAddRemove
        keep_dims_arg = op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        keep_dims_arg.i = 1

        del op.output[1:]
        del op.output_shape[1:]

    def convert_concat(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Concat.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = mge_op.params["axis"]

    def convert_batchnorm(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.BatchNorm.name

        gamma_value = get_symvar_value(
            self._bn_statistis_tensors[mge_op.inputs[1].name]
        ).flatten()
        beta_value = get_symvar_value(
            self._bn_statistis_tensors[mge_op.inputs[2].name]
        ).flatten()
        mean_value = get_symvar_value(mge_op.inputs[3]).flatten()
        var_value = get_symvar_value(mge_op.inputs[4]).flatten()
        epsilon_value = 1e-5

        scale_name = mge_op.name + "_scale"
        offset_name = mge_op.name + "_offset"
        scale_value = (1.0 / np.vectorize(math.sqrt)(
            var_value + epsilon_value)) * gamma_value
        offset_value = (-mean_value * scale_value) + beta_value
        self.add_tensor(
            scale_name, scale_value.shape, mace_pb2.DT_FLOAT, scale_value
        )
        self.add_tensor(
            offset_name, offset_value.shape, mace_pb2.DT_FLOAT, offset_value
        )
        self._skip_tensors.update([inp.name for inp in mge_op.inputs][1:])

        del op.input[1:]
        op.input.extend([scale_name, offset_name])
        # outputs[4] is the correct output
        del op.output[-1:]
        del op.output_shape[-1:]
        del op.output[:4]
        del op.output_shape[:4]

    def convert_shape(self, mge_op):
        op = self.convert_general_op(mge_op)
        op.type = MaceOp.Shape.name
        op.output_type.extend([mace_pb2.DT_INT32])

    # axis of subtensor should be constant
    # subtensor in megengine: numpy-like indexing
    def convert_subtensor(self, mge_op):
        op1 = self.convert_general_op(mge_op)
        op1.type = MaceOp.StridedSlice.name

        axis = mge_op.inputs[1].inferred_value
        t_shape = list(mge_op.inputs[0].imm_shape)

        begin_tensor_name = mge_op.name + "_begin"
        end_tensor_name = mge_op.name + "_end"
        stride_tensor_name = mge_op.name + "_stride"
        begin_tensor_shape = (len(t_shape),)
        end_tensor_shape = (len(t_shape),)
        stride_tensor_shape = (len(t_shape),)

        begin_vals = [0] * len(t_shape)
        end_vals = [shapei for shapei in t_shape]
        stride_vals = [1] * len(t_shape)

        def check_val(sym_var):
            try:
                val = sym_var.inferred_value[0]
            except TypeError:
                mace_check(
                    False, "you should feed const values for subtensor axis"
                )
            return val

        squeeze_dims = []
        idx = len(mge_op.inputs) - 1
        while idx:
            val = check_val(mge_op.inputs[idx])
            for ai in mge_op.params[::-1]:
                ai_idx = ai["axis"]
                if ai["step"] > 0:
                    stride_vals[ai_idx] = val
                    idx -= 1
                    if idx == 0:
                        break
                    val = check_val(mge_op.inputs[idx])
                if ai["end"] > 0:
                    if val < 0:
                        val = t_shape[ai_idx] + val
                    end_vals[ai_idx] = val
                    idx -= 1
                    if idx == 0:
                        break
                    val = check_val(mge_op.inputs[idx])
                if ai["begin"] > 0:
                    if val < 0:
                        val = t_shape[ai_idx] + val
                    begin_vals[ai_idx] = val
                    idx -= 1
                    if idx == 0:
                        break
                    val = check_val(mge_op.inputs[idx])
                if ai["idx"] > 0:
                    if val < 0:
                        val = t_shape[ai_idx] + val
                    squeeze_dims.append(ai_idx)
                    begin_vals[ai_idx] = val
                    end_vals[ai_idx] = val + 1
                    idx -= 1
                    if idx == 0:
                        break
                    val = check_val(mge_op.inputs[idx])

        for ai_idx in range(len(t_shape)):
            t_shape[ai_idx] = math.ceil(
                (end_vals[ai_idx] - begin_vals[ai_idx]) / stride_vals[ai_idx]
            )

        self.add_tensor(
            begin_tensor_name,
            begin_tensor_shape,
            mace_pb2.DT_INT32,
            begin_vals,
        )
        self.add_tensor(
            end_tensor_name, end_tensor_shape, mace_pb2.DT_INT32, end_vals
        )
        self.add_tensor(
            stride_tensor_name,
            stride_tensor_shape,
            mace_pb2.DT_INT32,
            stride_vals,
        )

        del op1.input[1:]
        op1.input.extend(
            [begin_tensor_name, end_tensor_name, stride_tensor_name]
        )

        if len(squeeze_dims) > 0:
            # create squeeze op to remove shape=1 dims
            mid_output_name = mge_op.name + "_mid_reshape"

            del op1.output[0]
            op1.output.extend([mid_output_name])
            output_shape = op1.output_shape[0]
            del output_shape.dims[:]
            output_shape.dims.extend(t_shape)

            op2 = self._mace_net_def.op.add()
            op2.type = MaceOp.Squeeze.name
            op2.name = mge_op.name + "_squeeze"

            data_type_arg = op2.arg.add()
            data_type_arg.name = "T"
            data_type_arg.i = self._option.data_type

            framework_type_arg = op2.arg.add()
            framework_type_arg.name = MaceKeyword.mace_framework_type_str
            framework_type_arg.i = FrameworkType.MEGENGINE.value

            ConverterUtil.add_data_format_arg(op2, DataFormat.NCHW)

            op2.input.extend([mid_output_name])
            op2.output.extend([mge_op.outputs[0].name])
            output_shape = op2.output_shape.add()
            output_shape.dims.extend(mge_op.outputs[0].imm_shape)

            axis_arg = op2.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            axis_arg.ints.extend(squeeze_dims)
