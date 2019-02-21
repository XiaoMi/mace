/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
HTA_DEF_OP(INPUT)
HTA_DEF_OP(OUTPUT)
HTA_DEF_OP(Nop)
HTA_DEF_OP(Const)
HTA_DEF_OP(Check)
HTA_DEF_OP(Close_f)
HTA_DEF_OP(Close_quint8)
HTA_DEF_OP(Close_q_quint8)
HTA_DEF_OP(Close_int32)
HTA_DEF_OP(Close_qint32)
HTA_DEF_OP(PPrint_8)
HTA_DEF_OP(PPrint_32)
HTA_DEF_OP(PPrint_f)
HTA_DEF_OP(PreFree)
HTA_DEF_OP(Flatten)

#ifndef HTA_DEF_OP_WREF
#define HTA_DEF_OP_WREF(NAME) HTA_DEF_OP(NAME) HTA_DEF_OP(NAME##_ref)
#define __SELF_HTA_DEF_OP_WREF
#endif

HTA_DEF_OP_WREF(QuantizedConv2d_8x8to32)
HTA_DEF_OP_WREF(QuantizedMatMul_8x8to32)
HTA_DEF_OP_WREF(QuantizeDownAndShrinkRange_32to8)
HTA_DEF_OP_WREF(QuantizedRelu_8)
HTA_DEF_OP_WREF(QuantizedReluX_8)
HTA_DEF_OP_WREF(QuantizedMaxPool_8)
HTA_DEF_OP_WREF(QuantizedAvgPool_8)
HTA_DEF_OP_WREF(QuantizedL2Pool_8)
HTA_DEF_OP_WREF(QuantizedConcat_8)
HTA_DEF_OP_WREF(QuantizedBiasAdd_8p8to32)
HTA_DEF_OP_WREF(Min_f)
HTA_DEF_OP_WREF(Max_f)
HTA_DEF_OP_WREF(Quantize)
HTA_DEF_OP_WREF(Dequantize)
HTA_DEF_OP_WREF(Supernode_8x8p8to8)

HTA_DEF_OP(QuantizedFlatten)
HTA_DEF_OP(Softmax_f)
HTA_DEF_OP(Conv2d_f)
HTA_DEF_OP(MatMul_f)
HTA_DEF_OP(Relu_f)
HTA_DEF_OP(ReluX_f)
HTA_DEF_OP(AvgPool_f)
HTA_DEF_OP(L2Pool_f)
HTA_DEF_OP(MaxPool_f)
HTA_DEF_OP(Concat_f)
HTA_DEF_OP(BiasAdd_f)
HTA_DEF_OP(LRN_f)

HTA_DEF_OP(Variable)
HTA_DEF_OP(Assign)
HTA_DEF_OP(Reshape)
HTA_DEF_OP(QuantizedReshape)
HTA_DEF_OP(Tanh_f)
HTA_DEF_OP(Sigmoid_f)
HTA_DEF_OP(Slice_8)
HTA_DEF_OP(Slice_f)
HTA_DEF_OP(QuantizedSlice_8)
HTA_DEF_OP(Add_f)
HTA_DEF_OP(Mul_f)
HTA_DEF_OP(Minimum_f)
HTA_DEF_OP(Maximum_f)

HTA_DEF_OP_WREF(Requantize_32to8)
HTA_DEF_OP_WREF(RequantizationRange_32)

HTA_DEF_OP(Neg_f)
HTA_DEF_OP(Sub_f)
HTA_DEF_OP(AddN_f)
HTA_DEF_OP(Range_int32)
HTA_DEF_OP(Rank_int32)
HTA_DEF_OP(Transpose_int32)
HTA_DEF_OP(Transpose_f)
HTA_DEF_OP(InstanceNorm_f)
HTA_DEF_OP_WREF(QuantizedInstanceNorm_8)
HTA_DEF_OP(Sub_int32)
HTA_DEF_OP(Add_int32)
HTA_DEF_OP(Split_f)
HTA_DEF_OP(Dequantize_qint32_f)
HTA_DEF_OP(PRelu_f)
HTA_DEF_OP_WREF(QuantizedPRelu_8)
HTA_DEF_OP(Sum_f)
HTA_DEF_OP(Prod_f)
HTA_DEF_OP(Mul_int32)
HTA_DEF_OP(LogicalAnd_int32)
HTA_DEF_OP(LogicalOr_int32)
HTA_DEF_OP(LogicalXor_int32)
HTA_DEF_OP(Shape_int32)
HTA_DEF_OP(Pack_int32)
HTA_DEF_OP(MirrorPad_f)
HTA_DEF_OP(ResizeNearestNeighbor_f)
HTA_DEF_OP(StridedSlice_int32)
HTA_DEF_OP(StridedSlice_f)
HTA_DEF_OP(ExpandDims_int32)
HTA_DEF_OP(ExpandDims_f)

HTA_DEF_OP(LogSoftmax_f)
HTA_DEF_OP(Split_int32)
HTA_DEF_OP(QuantizedSplit_8)

HTA_DEF_OP(Deconv_f)
HTA_DEF_OP_WREF(QuantizedDeconv_8x8to32)

HTA_DEF_OP_WREF(QuantizedMul_8x8to32)
HTA_DEF_OP_WREF(QuantizedAdd_8p8to32)
HTA_DEF_OP_WREF(QuantizedSigmoid_8)
HTA_DEF_OP_WREF(QuantizedTanh_8)
HTA_DEF_OP_WREF(QuantizedSoftmax_8)
HTA_DEF_OP_WREF(QuantizedLRN_8)
HTA_DEF_OP_WREF(Quantizedpad2d_frame_8p)
HTA_DEF_OP_WREF(QuantizedSub_8p8to32)
HTA_DEF_OP_WREF(QuantizedMaximum_8)
HTA_DEF_OP_WREF(QuantizedMinimum_8)

HTA_DEF_OP(Pad_f)
HTA_DEF_OP(SpaceToBatchND_f)
HTA_DEF_OP(BatchToSpaceND_f)
HTA_DEF_OP(QuantizedPad_8)
HTA_DEF_OP(ResizeBilinear_f)
HTA_DEF_OP(ConcatV2_f)
HTA_DEF_OP(ConcatV2_int32)
HTA_DEF_OP(Prod_int32)
HTA_DEF_OP(Slice_int32)

HTA_DEF_OP(QuantizedAdd_8p8to8)
HTA_DEF_OP(QuantizedResizeBilinear_8)
HTA_DEF_OP(Supernode_8x8p8to8_d32)
HTA_DEF_OP(Convert_to_d32)
HTA_DEF_OP(Convert_from_d32)
HTA_DEF_OP_WREF(QuantizedMaxPool_8_d32)
HTA_DEF_OP_WREF(QuantizedConcat_8_d32)
HTA_DEF_OP_WREF(QuantizedAvgPool_8_d32)

HTA_DEF_OP(Sink)

HTA_DEF_OP_WREF(QuantizedPRelu_8_d32)
HTA_DEF_OP_WREF(AutoQuantize)
HTA_DEF_OP_WREF(QuantizedDepthwiseConv2d_8x8to32)
HTA_DEF_OP_WREF(DepthwiseConv2d_f)
HTA_DEF_OP(DepthwiseSupernode_8x8p8to8)
HTA_DEF_OP(DepthwiseSupernode_8x8p8to8_d32)

HTA_DEF_OP_WREF(QuantizedMul_8x8to8_d32)

HTA_DEF_OP(FullyConnected_u8)
#if 0
HTA_DEF_OP_WREF(QuantizedFC_8x8p8to8)
#endif

HTA_DEF_OP_WREF(QuantizedAdd_8p8to8_d32)

HTA_DEF_OP_WREF(QuantizedClamp_8)
HTA_DEF_OP(Clamp_f)
HTA_DEF_OP(QuantizeForTest_d32)
HTA_DEF_OP(Close_d32)
HTA_DEF_OP_WREF(QuantizedSub_8p8to8_d32)

HTA_DEF_OP(InputSupernode_8x8p8to8_outd32)
HTA_DEF_OP(QuantizedLRN_8_d32)
HTA_DEF_OP_WREF(QuantizedBiasAdd_32p32to32)
HTA_DEF_OP_WREF(Quantize_int32)

HTA_DEF_OP(Supernode_8x8p32to8)
HTA_DEF_OP(DepthwiseSupernode_8x8p32to8)
HTA_DEF_OP(Supernode_8x8p32to8_d32)
HTA_DEF_OP(DepthwiseSupernode_8x8p32to8_d32)
HTA_DEF_OP(InputSupernode_8x8p32to8_outd32)

HTA_DEF_OP(PPrint_8_d32)
HTA_DEF_OP(PPrintWithPadding_8_d32)
HTA_DEF_OP_WREF(AutoQuantize_d32)

HTA_DEF_OP_WREF(QuantizedTanh_8_d32)
HTA_DEF_OP_WREF(QuantizedSigmoid_8_d32)
HTA_DEF_OP_WREF(QuantizedSoftmax_8_d32)


HTA_DEF_OP_WREF(QuantizedL2Pool_8_d32)

HTA_DEF_OP(Gather_f)
HTA_DEF_OP(Gather_int32)
HTA_DEF_OP(Gather_8)
HTA_DEF_OP(Table_f)
HTA_DEF_OP(Table_int32)
HTA_DEF_OP(Table_8)

HTA_DEF_OP(FillPadding_8_d32)
HTA_DEF_OP(QuantizedResizeBilinear_8_d32)

HTA_DEF_OP(QuantizeINPUT_f_to_8)
HTA_DEF_OP_WREF(DeconvBias_8x8to32)

HTA_DEF_OP(SpaceToBatchND_8)
HTA_DEF_OP(BatchToSpaceND_8)


HTA_DEF_OP(SpaceToDepth_f)
HTA_DEF_OP(DepthToSpace_f)
HTA_DEF_OP(SpaceToDepth_8)
HTA_DEF_OP(DepthToSpace_8)

HTA_DEF_OP(DequantizeOUTPUT_8tof)
HTA_DEF_OP(QuantizedBatchNorm_8x8p8to8)
HTA_DEF_OP(QuantizedBatchNorm_8x8p32to8)
HTA_DEF_OP(QuantizedBatchNorm_8x8p8to8_d32)
HTA_DEF_OP(QuantizedBatchNorm_8x8p32to8_d32)

HTA_DEF_OP_WREF(QuantizedInstanceNorm_8_d32)
HTA_DEF_OP_WREF(QuantizedInstanceNormBG_8)
HTA_DEF_OP_WREF(QuantizedInstanceNormBG_8_d32)

HTA_DEF_OP(SuperFC_8x8p32to8)
HTA_DEF_OP(SuperFC_8x8p32to8_ref)
HTA_DEF_OP(SuperFC_8x8p32to8_d32)

HTA_DEF_OP(ChannelShuffle_f)
HTA_DEF_OP(ChannelShuffle_int32)
HTA_DEF_OP_WREF(QuantizedChannelShuffle_8)
HTA_DEF_OP(QuantizedChannelShuffle_8_d32)
/* this is in op_chanshuffle_d32.c*/
HTA_DEF_OP(QuantizedSplit_8_d32)

HTA_DEF_OP(QuantizedCrop_8)
HTA_DEF_OP(ResizeUnitSquare_f)
HTA_DEF_OP_WREF(ResizeUnitSquare_8)
HTA_DEF_OP_WREF(Nv21ToRgb_8)
HTA_DEF_OP_WREF(RgbaToRgb_8)
HTA_DEF_OP_WREF(Argb32ToRgb_8)
HTA_DEF_OP(Permute_f)
HTA_DEF_OP(QuantizedPermute_8)
HTA_DEF_OP_WREF(QuantizedRoiPool_8)
HTA_DEF_OP(Proposal_f)
HTA_DEF_OP(RoiAlign_f)
HTA_DEF_OP_WREF(QuantizedRoiAlign_8)
HTA_DEF_OP_WREF(Implode_8)
HTA_DEF_OP(QuantizedConcat_8_nond32)

HTA_DEF_OP(Close_16tof)
HTA_DEF_OP(QuantizedLstmInput_16x16to16)
HTA_DEF_OP(QuantizedLstmOutput_16x16to8)

HTA_DEF_OP(Quantize_16)
HTA_DEF_OP(Dequantize_16)
HTA_DEF_OP(Convert_8_16)
HTA_DEF_OP(QuantizedTanh_16)
HTA_DEF_OP(QuantizedSigmoid_16)

HTA_DEF_OP_WREF(QuantizeDownAndShrinkRange_32to16)
HTA_DEF_OP_WREF(Requantize_32to16)
HTA_DEF_OP_WREF(QuantizedMatMul_8x8p32to16)

HTA_DEF_OP(QuantizedStridedSlice_8)
HTA_DEF_OP(Bbox_Transform_f)
HTA_DEF_OP(Softmax_uint8)

HTA_DEF_OP(QuantizedFakeConcat_8_d32)

HTA_DEF_OP(DepthToSpace_8_d32)
HTA_DEF_OP(OemNode)

HTA_DEF_OP(QuantizedPad_8_d32)
// Add new operations above this line
#ifdef __SELF_HTA_DEF_OP_WREF
#undef __SELF_HTA_DEF_OP_WREF
#undef HTA_DEF_OP_WREF
#endif

