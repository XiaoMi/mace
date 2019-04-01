/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

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
// NOLINT(build/header_guard)

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
DEF_OP_WREF(QuantizedL2Pool_8)
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
DEF_OP(L2Pool_f)
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
DEF_OP_WREF(Quantizedpad2d_frame_8p)
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
DEF_OP(QuantizedResizeBilinear_8)
DEF_OP(Supernode_8x8p8to8_d32)
DEF_OP(Convert_to_d32)
DEF_OP(Convert_from_d32)
DEF_OP_WREF(QuantizedMaxPool_8_d32)
DEF_OP_WREF(QuantizedConcat_8_d32)
DEF_OP_WREF(QuantizedAvgPool_8_d32)

DEF_OP(Sink)

DEF_OP_WREF(QuantizedPRelu_8_d32)
DEF_OP_WREF(AutoQuantize)
DEF_OP_WREF(QuantizedDepthwiseConv2d_8x8to32)
DEF_OP(QuantizedTransposeConv2d_8x8p32to8)
DEF_OP_WREF(DepthwiseConv2d_f)
DEF_OP(DepthwiseSupernode_8x8p8to8)
DEF_OP(DepthwiseSupernode_8x8p8to8_d32)

DEF_OP_WREF(QuantizedMul_8x8to8_d32)

DEF_OP(FullyConnected_u8)
#if 0
    DEF_OP_WREF(QuantizedFC_8x8p8to8)
#endif

DEF_OP_WREF(QuantizedAdd_8p8to8_d32)

DEF_OP_WREF(QuantizedClamp_8)
DEF_OP(Clamp_f)
DEF_OP(QuantizeForTest_d32)
DEF_OP(Close_d32)
DEF_OP_WREF(QuantizedSub_8p8to8_d32)

DEF_OP(InputSupernode_8x8p8to8_outd32)
DEF_OP(QuantizedLRN_8_d32)
DEF_OP_WREF(QuantizedBiasAdd_32p32to32)
DEF_OP_WREF(Quantize_int32)

DEF_OP(Supernode_8x8p32to8)
DEF_OP(DepthwiseSupernode_8x8p32to8)
DEF_OP(Supernode_8x8p32to8_d32)
DEF_OP(DepthwiseSupernode_8x8p32to8_d32)
DEF_OP(InputSupernode_8x8p32to8_outd32)

DEF_OP(PPrint_8_d32)
DEF_OP(PPrintWithPadding_8_d32)
DEF_OP_WREF(AutoQuantize_d32)

DEF_OP_WREF(QuantizedTanh_8_d32)
DEF_OP_WREF(QuantizedSigmoid_8_d32)
DEF_OP_WREF(QuantizedSoftmax_8_d32)


DEF_OP_WREF(QuantizedL2Pool_8_d32)

DEF_OP(Gather_f)
DEF_OP(Gather_int32)
DEF_OP(Gather_8)
DEF_OP(Table_f)
DEF_OP(Table_int32)
DEF_OP(Table_8)

DEF_OP(FillPadding_8_d32)
DEF_OP(QuantizedResizeBilinear_8_d32)

DEF_OP(QuantizeINPUT_f_to_8)
DEF_OP_WREF(DeconvBias_8x8to32)

DEF_OP(SpaceToBatchND_8)
DEF_OP(BatchToSpaceND_8)


DEF_OP(SpaceToDepth_f)
DEF_OP(DepthToSpace_f)
DEF_OP(SpaceToDepth_8)
DEF_OP(DepthToSpace_8)

DEF_OP(DequantizeOUTPUT_8tof)
DEF_OP(QuantizedBatchNorm_8x8p8to8)
DEF_OP(QuantizedBatchNorm_8x8p32to8)
DEF_OP(QuantizedBatchNorm_8x8p8to8_d32)
DEF_OP(QuantizedBatchNorm_8x8p32to8_d32)

DEF_OP_WREF(QuantizedInstanceNorm_8_d32)
DEF_OP_WREF(QuantizedInstanceNormBG_8)
DEF_OP_WREF(QuantizedInstanceNormBG_8_d32)

DEF_OP(SuperFC_8x8p32to8)
DEF_OP(SuperFC_8x8p32to8_ref)
DEF_OP(SuperFC_8x8p32to8_d32)

DEF_OP(ChannelShuffle_f)
DEF_OP(ChannelShuffle_int32)
DEF_OP_WREF(QuantizedChannelShuffle_8)
DEF_OP(QuantizedChannelShuffle_8_d32)
/* this is in op_chanshuffle_d32.c*/
DEF_OP(QuantizedSplit_8_d32)

DEF_OP(QuantizedCrop_8)
DEF_OP(ResizeUnitSquare_f)
DEF_OP_WREF(ResizeUnitSquare_8)
DEF_OP_WREF(Nv21ToRgb_8)
DEF_OP_WREF(RgbaToRgb_8)
DEF_OP_WREF(Argb32ToRgb_8)
DEF_OP(Permute_f)
DEF_OP(QuantizedPermute_8)
DEF_OP_WREF(QuantizedRoiPool_8)
DEF_OP(Proposal_f)
DEF_OP(RoiAlign_f)
DEF_OP_WREF(QuantizedRoiAlign_8)
DEF_OP_WREF(Implode_8)
DEF_OP(QuantizedConcat_8_nond32)

DEF_OP(Close_16tof)
DEF_OP(QuantizedLstmInput_16x16to16)
DEF_OP(QuantizedLstmOutput_16x16to8)

DEF_OP(Quantize_16)
DEF_OP(Dequantize_16)
DEF_OP(Convert_8_16)
DEF_OP(QuantizedTanh_16)
DEF_OP(QuantizedSigmoid_16)

DEF_OP_WREF(QuantizeDownAndShrinkRange_32to16)
DEF_OP_WREF(Requantize_32to16)
DEF_OP_WREF(QuantizedMatMul_8x8p32to16)

DEF_OP(QuantizedStridedSlice_8)
DEF_OP(Bbox_Transform_f)
DEF_OP(Softmax_uint8)

DEF_OP(QuantizedFakeConcat_8_d32)

DEF_OP(DepthToSpace_8_d32)
DEF_OP(OemNode)

DEF_OP(QuantizedPad_8_d32)

DEF_OP(QuantizedSqrt_8)
DEF_OP(QuantizedSlice_16)
DEF_OP(QuantizedMin_8)
DEF_OP(QuantizedMax_8)

DEF_OP(Transpose_8)

DEF_OP(Close_u16tof)
DEF_OP(QuantizeForTest_16b_d32)
DEF_OP(QuantizeForTest_u16b_d32)
DEF_OP(Close_16b_d32)
DEF_OP(Close_u16b_d32)
DEF_OP(Convert_from_d32_b16)
DEF_OP(Supernode_16x16p16to16_d32)
DEF_OP(Supernode_u16x16p16to16_d32)

DEF_OP(QuantizedMatMulDims_8x8p32to16)
DEF_OP(BatchSeqConfig)
DEF_OP(QuantizedDiv_8)
DEF_OP(QuantizedRecip_8)

DEF_OP(QuantizedNeg_8)
DEF_OP(QuantizedNeg_8_d32)
DEF_OP(Neg_int32)
DEF_OP(Abs_f)
DEF_OP(Abs_int32)

DEF_OP(QuantizedSub_8p8to8)
DEF_OP(Box_Decoder_f)
DEF_OP(QuantizedExtractGlimpse_8)
DEF_OP(QuantizedTile_8)

DEF_OP_WREF(QuantizedMul_8x8to8)
DEF_OP(QuantizedSum_8to32)
DEF_OP(ImageTransform_f)

DEF_OP(Convert_to_aix_d32)
DEF_OP(Convert_from_aix)
DEF_OP(Convert_from_aix_d32)

DEF_OP(MultiClassNms_f)

DEF_OP(BatchToSpaceND_8_d32)
DEF_OP(SpaceToBatchND_8_d32)
DEF_OP(Supernode3322_8x8p8to8)
DEF_OP(Supernode3322_8x8p32to8)

DEF_OP(Convert_int32_f)
DEF_OP(ArgMax_ftoInt32)
DEF_OP(ArgMax_8toInt32)
DEF_OP(ArgMax_8)
DEF_OP(Supernode_8x8p32to8_ref)
DEF_OP(HeatmapMaxKP_f)
DEF_OP(TopK_f)
DEF_OP(TopK_8)
DEF_OP(CastFloat32ToInt32)
DEF_OP(CastFloat32ToUInt8)
DEF_OP(CastInt32ToFloat32)
DEF_OP(CastInt32ToUInt8)
DEF_OP(CastUInt8ToFloat32)
DEF_OP(CastUInt8ToInt32)
DEF_OP(AxisShuffle_8)
DEF_OP(ResizeNearestNeighbor_8)
DEF_OP(QuantizedHeatmapMaxKP_8)
DEF_OP(Moments_8to32)
DEF_OP(ArgMin_8)
DEF_OP(Select_f)
DEF_OP(Select_8)
DEF_OP(QuantizedGroupedConv2d_8x8p32to8)

// Add new operations above this line
#ifdef __SELF_DEF_OP_WREF
#undef __SELF_DEF_OP_WREF
#undef DEF_OP_WREF
#endif
