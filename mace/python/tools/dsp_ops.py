
class DspOps(object):
  def __init__(self):
    self.dsp_ops = {
      'INPUT': 'INPUT"',
      'OUTPUT': 'OUTPUT',
      'NoOp': 'Nop',
      'FLATTEN': 'Flatten',
      'Identity': 'Nop',
      'Placeholder': 'INPUT',
      'Const': 'Const',
      'QuantizedConv2D': 'QuantizedConv2d_8x8to32',
      'QuantizedMatMul': 'QuantizedMatMul_8x8to32',
      'QuantizeDownAndShrinkRange': 'QuantizeDownAndShrinkRange_32to8',
      'QuantizedRelu': 'QuantizedRelu_8',
      'QuantizedReluX': 'QuantizedReluX_8',
      'QuantizedMaxPool': 'QuantizedMaxPool_8',
      'QuantizedAvgPool': 'QuantizedAvgPool_8',
      'QuantizedConcat': 'QuantizedConcat_8',
      'QuantizedBiasAdd': 'QuantizedBiasAdd_8p8to32',
      'Min': 'Min_f',
      'Max': 'Max_f',
      'QuantizeV2': 'Quantize',
      'Dequantize': 'Dequantize',
      'Softmax': 'Softmax_f',
      'Reshape': 'Reshape',
      'QuantizedReshape': 'QuantizedReshape',
      'Sigmoid': 'Sigmoid_f',
      'Slice': 'Slice_f',
      'Add': 'Add_f',
      'Mul': 'Mul_f',
      'Requantize': 'Requantize_32to8',
      'RequantizationRange': 'RequantizationRange_32',
      'Sub': 'Sub_f',
      'Pack': 'Pack_int32',
      'StridedSlice': 'StridedSlice_f',
      'ExpandDims': 'ExpandDims_f',
      'QuantizedMul': 'QuantizedMul_8x8to32',
      'QuantizedAdd': 'QuantizedAdd_8p8to32',
      'Pad': 'Pad_f',
      'SpaceToBatchND': 'SpaceToBatchND_f',
      'BatchToSpaceND': 'BatchToSpaceND_f',
      'ResizeBilinear': 'ResizeBilinear_f',
      'ConcatV2': 'ConcatV2_f',
      'Conv2DBackpropInput': 'Deconv_f',
      'Tanh': 'Tanh_f',
      'Split': 'Split_f',
      'Transpose': 'Transpose_f',
      'Concat': 'Concat_f',
      'AddN': 'AddN_f',
    }
  def has_op(self, tf_op):
    return tf_op in self.dsp_ops

  def map_nn_op(self, tf_op):
    if tf_op not in self.dsp_ops:
      raise Exception('Could not map nn op for: ', tf_op)
    return self.dsp_ops[tf_op]


