/*
 * You probably want to
 *
 *    ##    #####   #####
 *   #  #   #    #  #    #
 *  #    #  #    #  #    #
 *  ######  #    #  #    #
 *  #    #  #    #  #    #
 *  #    #  #####   #####
 *
 *
 *  #    #   ####   #####   ######   ####
 *  ##   #  #    #  #    #  #       #
 *  # #  #  #    #  #    #  #####    ####
 *  #  # #  #    #  #    #  #            #
 *  #   ##  #    #  #    #  #       #    #
 *  #    #   ####   #####   ######   ####
 *
 *
 *    ##     #####
 *   #  #      #
 *  #    #     #
 *  ######     #
 *  #    #     #
 *  #    #     #
 *
 *
 *   #####  #    #  ######
 *     #    #    #  #
 *     #    ######  #####
 *     #    #    #  #
 *     #    #    #  #
 *     #    #    #  ######
 *
 *
 *  ######  #    #  #####
 *  #       ##   #  #    #
 *  #####   # #  #  #    #
 *  #       #  # #  #    #
 *  #       #   ##  #    #
 *  ######  #    #  #####
 *
 * otherwise the interface becomes incompatible.
 */
DEF_OP(INPUT)
DEF_OP(OUTPUT)
DEF_OP(Nop)
DEF_OP(Const)
DEF_OP(Check)
DEF_OP(Close_f)
DEF_OP(Close_quint8)
DEF_OP(Close_q_quint8)
DEF_OP(Close_int32)
DEF_OP(Close_qint32)
DEF_OP(PPrint_8)
DEF_OP(PPrint_32)
DEF_OP(PPrint_f)
DEF_OP(PreFree)
DEF_OP(Flatten)

#ifndef DEF_OP_WREF
#define DEF_OP_WREF(NAME) DEF_OP(NAME) DEF_OP(NAME##_ref)
#define __SELF_DEF_OP_WREF
#endif

DEF_OP_WREF(QuantizedConv2d_8x8to32)
DEF_OP_WREF(QuantizedMatMul_8x8to32)
DEF_OP_WREF(QuantizeDownAndShrinkRange_32to8)
DEF_OP_WREF(QuantizedRelu_8)
DEF_OP_WREF(QuantizedReluX_8)
DEF_OP_WREF(QuantizedMaxPool_8)
DEF_OP_WREF(QuantizedAvgPool_8)
DEF_OP_WREF(QuantizedConcat_8)
DEF_OP_WREF(QuantizedBiasAdd_8p8to32)
DEF_OP_WREF(Min_f)
DEF_OP_WREF(Max_f)
DEF_OP_WREF(Quantize)
DEF_OP_WREF(Dequantize)
DEF_OP_WREF(Supernode_8x8p8to8)

DEF_OP(QuantizedFlatten)
DEF_OP(Softmax_f)
DEF_OP(Conv2d_f)
DEF_OP(MatMul_f)
DEF_OP(Relu_f)
DEF_OP(ReluX_f)
DEF_OP(AvgPool_f)
DEF_OP(MaxPool_f)
DEF_OP(Concat_f)
DEF_OP(BiasAdd_f)
DEF_OP(LRN_f)

DEF_OP(Variable)
DEF_OP(Assign)
DEF_OP(Reshape)
DEF_OP(QuantizedReshape)
DEF_OP(Tanh_f)
DEF_OP(Sigmoid_f)
DEF_OP(Slice_8)
DEF_OP(Slice_f)
DEF_OP(QuantizedSlice_8)
DEF_OP(Add_f)
DEF_OP(Mul_f)
DEF_OP(Minimum_f)
DEF_OP(Maximum_f)

DEF_OP_WREF(Requantize_32to8)
DEF_OP_WREF(RequantizationRange_32)

DEF_OP(Neg_f)
DEF_OP(Sub_f)
DEF_OP(AddN_f)
DEF_OP(Range_int32)
DEF_OP(Rank_int32)
DEF_OP(Transpose_int32)
DEF_OP(Transpose_f)
DEF_OP(InstanceNorm_f)
DEF_OP_WREF(QuantizedInstanceNorm_8)
DEF_OP(Sub_int32)
DEF_OP(Add_int32)
DEF_OP(Split_f)
DEF_OP(Dequantize_qint32_f)
DEF_OP(PRelu_f)
DEF_OP_WREF(QuantizedPRelu_8)
DEF_OP(Sum_f)
DEF_OP(Prod_f)
DEF_OP(Mul_int32)
DEF_OP(LogicalAnd_int32)
DEF_OP(LogicalOr_int32)
DEF_OP(LogicalXor_int32)
DEF_OP(Shape_int32)
DEF_OP(Pack_int32)
DEF_OP(MirrorPad_f)
DEF_OP(ResizeNearestNeighbor_f)
DEF_OP(StridedSlice_int32)
DEF_OP(StridedSlice_f)
DEF_OP(ExpandDims_int32)
DEF_OP(ExpandDims_f)

DEF_OP(LogSoftmax_f)
DEF_OP(Split_int32)
DEF_OP(QuantizedSplit_8)

DEF_OP(Deconv_f)
DEF_OP_WREF(QuantizedDeconv_8x8to32)

DEF_OP_WREF(QuantizedMul_8x8to32)
DEF_OP_WREF(QuantizedAdd_8p8to32)
DEF_OP_WREF(QuantizedSigmoid_8)
DEF_OP_WREF(QuantizedTanh_8)
DEF_OP_WREF(QuantizedSoftmax_8)
DEF_OP_WREF(QuantizedLRN_8)
DEF_OP_WREF(QuantizedSub_8p8to32)
DEF_OP_WREF(QuantizedMaximum_8)
DEF_OP_WREF(QuantizedMinimum_8)

DEF_OP(Pad_f)
DEF_OP(SpaceToBatchND_f)
DEF_OP(BatchToSpaceND_f)
DEF_OP(QuantizedPad_8)
DEF_OP(ResizeBilinear_f)
DEF_OP(ConcatV2_f)
DEF_OP(ConcatV2_int32)
DEF_OP(Prod_int32)
DEF_OP(Slice_int32)

DEF_OP(QuantizedAdd_8p8to8)

DEF_OP_WREF(AutoQuantize)
DEF_OP_WREF(QuantizedDepthwiseConv2d_8x8to32)
DEF_OP(DepthwiseConv2d_f)

#ifdef __SELF_DEF_OP_WREF
#undef __SELF_DEF_OP_WREF
#undef DEF_OP_WREF
#endif

