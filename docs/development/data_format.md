Data Format
===========

As we all know, input/output tensors in CNN model have data format like
`NHWC`(tensorflow) or `NCHW`(caffe), but there is no data format for non-CNN model.

However, in MACE, CNN model run on `CPU` with `float` type using `NCHW` data format,
while the others using `NHWC` data format.

To support all models, so there are some concepts in `MACE` you should know.

Source Data Format
-----------------------
Source Data Format(`src_df` for short) stands for the original data format where 
the model come from. For example, if you use caffe, the `src_df` is `NCHW`.
We need this data format because some operators(Reshape etc.) are 
 related to the data format.


Operators Partition
--------------------
Generally, operators could be divided into 2 categories
based on whether the operator needs inputs with fixed data format(`NHWC` or `NCHW`),
one is the operators whose inputs have fixed data format(like `convolution`),
the other is the operators whose inputs should be the same with source framework.

Since the data format the operators need in MACE may be inconsistent with the original framework,
 we need to add `Transpose` operator to transpose the input tensors if necessary. 
 
However, for some operators like `concat`,
we could transpose their arguments to eliminate `Transpose` op for acceleration.

Based on these conditions, We partition the ops into 3 categories.
1. Ops with fixed inputs' data format(`FixedDataFormatOps`): `Convolution`, `Depthwise Convolution`, etc.
2. Ops could eliminate `Transpose` by transposing their arguments(`TransposableDataFormatOps`): `Concat`, `Element-wise`, etc.
3. Ops keeping consistent with source platform(`SourceDataFormatOps`): `Reshape`, `ExpandDims`, etc.

By default, the operators not in either `FixedDataFormatOps` or `TransposableDataFormatOps`
are listed in `SourceDataFormatOps`.

For detailed information, you could refer to [code](https://github.com/XiaoMi/mace/blob/master/mace/python/tools/converter_tool/base_converter.py).


Data Format in Operator
------------------------
Based on the operator partition strategy, every operator in `MACE` has 
data format argument which stands for the wanted inputs' data format,
the values could be one of the [`NHWC`, `NCHW`, `AUTO`].
1. `NHWC` or `NCHW` represent `src_df`.
2. `AUTO` represents the operator's inputs must have fixed data format,
and the real data format will be determined at runtime.
the data format of operators in `FixedDataFormatOps` must be `AUTO`,
while the data format of operators in `TransposableDataFormatOps` 
is determined based on their inputs' ops data format.
 
MACE will transpose the input tensors based on the data format information automatically at runtime.


Data Format of Model's Inputs/Outputs
-------------------------------------
1. If the model's inputs/outputs have data format, MACE supports the data format 
`NHWC` and `NCHW`.
2. If the model's inputs/outputs do not have data format, just set `NONE` for
 model's inputs and outputs at `model deployment file` and `MaceTensor`.
