Benchmark usage
===============

This part contains the usage of MACE benchmark tools.

Overview
--------

As mentioned in the previous part, there are two kinds of benchmark tools,
one for operator and the other for model.

Operator Benchmark
------------------

Operator Benchmark is used for test and optimize the performance of specific operator.

=====
Usage
=====

    .. code:: bash

        python tools/bazel_adb_run.py --target="//mace/ops:ops_benchmark" --run_target=True  --args="--filter=.*BM_CONV.*"

======
Output
======

    .. code:: bash

        Benchmark                                                    Time(ns) Iterations Input(MB/s)   GMACPS
        ------------------------------------------------------------------------------------------------------
        MACE_BM_CONV_2D_1_1024_7_7_K1x1S1D1_SAME_1024_float_CPU       1759129        479     114.09      29.21
        MACE_BM_CONV_2D_1_1024_7_7_K1x1S1D1_SAME_1024_float_GPU       4031301        226      49.79      12.75
        MACE_BM_CONV_2D_1_1024_7_7_K1x1S1D1_SAME_1024_half_GPU        3996357        266      25.11      12.86
        MACE_BM_CONV_2D_1_1024_7_7_K1x1S1D1_SAME_1024_uint8_t_CPU      914994       1093      54.84      56.15


===========
Explanation
===========

.. list-table::
    :header-rows: 1

    * - Options
      - Usage
    * - Benchmark
      - Benchmark unit name.
    * - Time
      - Time of one round.
    * - Iterations
      - the number of iterations to run, which is between 10 and 1000,000,000. the value is calculated based on the strategy total run time does not exceed 1s.
    * - Input
      - The bandwidth of dealing with input. the unit is MB/s.
    * - GMACPS
      - The speed of running MACs(multiply-accumulation). the unit is G/s.

Model Benchmark
---------------

Model Benchmark is used for test and optimize the performance of your model.
This tool could record the running time of the model and the detailed running information of each operator of your model.

=====
Usage
=====

    .. code:: bash

        python tools/converter.py benchmark --config=/path/to/your/model_deployment.yml

======
Output
======

    .. code:: bash

        I benchmark_model.cc:158 ---------------------------------------------------------------------
        I benchmark_model.cc:158                                Warm Up
        I benchmark_model.cc:158 ----------------------------------------------------------------------
        I benchmark_model.cc:158 | round | first(ms) | curr(ms) | min(ms) | max(ms) | avg(ms) |   std |
        I benchmark_model.cc:158 ----------------------------------------------------------------------
        I benchmark_model.cc:158 |     1 |    51.481 |   51.481 |  51.481 |  51.481 |  51.481 | 0.000 |
        I benchmark_model.cc:158 ----------------------------------------------------------------------
        I benchmark_model.cc:158
        I benchmark_model.cc:158 ------------------------------------------------------------------------
        I benchmark_model.cc:158                          Run without statistics
        I benchmark_model.cc:158 -------------------------------------------------------------------------
        I benchmark_model.cc:158 | round | first(ms) | curr(ms) | min(ms) | max(ms) | avg(ms) |      std |
        I benchmark_model.cc:158 -------------------------------------------------------------------------
        I benchmark_model.cc:158 |   100 |    30.272 |   31.390 |  29.938 |  45.966 |  30.913 | 1850.983 |
        I benchmark_model.cc:158 -------------------------------------------------------------------------
        I benchmark_model.cc:158
        I benchmark_model.cc:158 -----------------------------------------------------------------------
        I benchmark_model.cc:158                           Run with statistics
        I benchmark_model.cc:158 ------------------------------------------------------------------------
        I benchmark_model.cc:158 | round | first(ms) | curr(ms) | min(ms) | max(ms) | avg(ms) |     std |
        I benchmark_model.cc:158 ------------------------------------------------------------------------
        I benchmark_model.cc:158 |   100 |    32.358 |   33.327 |  32.293 |  33.607 |  33.002 | 310.435 |
        I benchmark_model.cc:158 ------------------------------------------------------------------------
        I statistics.cc:343 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343                                                                                      Sort by Run Order
        I statistics.cc:343 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343 |         Op Type |  Start | First | Avg(ms) |     % |    cdf% | GMACPS | Stride |   Pad |    Filter Shape |   Output Shape | Dilation |                                               name |
        I statistics.cc:343 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343 |       Transpose |  0.000 | 0.102 |   0.100 | 0.315 |   0.315 |  0.000 |        |       |                 |  [1,3,224,224] |          |                                              input |
        I statistics.cc:343 |          Conv2D |  0.107 | 1.541 |   1.570 | 4.943 |   5.258 |  6.904 |  [2,2] |  SAME |      [32,3,3,3] | [1,32,112,112] |    [1,1] |             MobilenetV1/MobilenetV1/Conv2d_0/Relu6 |
        I statistics.cc:343 | DepthwiseConv2d |  1.724 | 0.936 |   0.944 | 2.972 |   8.230 |  3.827 |  [1,1] |  SAME |      [1,32,3,3] | [1,32,112,112] |    [1,1] |   MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6 |
        I statistics.cc:343 |         Softmax | 32.835 | 0.039 |   0.042 | 0.131 |  99.996 |  0.000 |        |       |                 |       [1,1001] |          |                    MobilenetV1/Predictions/Softmax |
        I statistics.cc:343 |        Identity | 32.880 | 0.001 |   0.001 | 0.004 | 100.000 |  0.000 |        |       |                 |       [1,1001] |          | mace_output_node_MobilenetV1/Predictions/Reshape_1 |
        I statistics.cc:343 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343
        I statistics.cc:343 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343                                                                              Sort by Computation Time
        I statistics.cc:343 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343 | Op Type |  Start | First | Avg(ms) |     % |   cdf% | GMACPS | Stride |  Pad |    Filter Shape |   Output Shape | Dilation |                                              name |
        I statistics.cc:343 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343 |  Conv2D | 30.093 | 2.102 |   2.198 | 6.922 |  6.922 | 23.372 |  [1,1] | SAME | [1024,1024,1,1] |   [1,1024,7,7] |    [1,1] | MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D |  7.823 | 2.115 |   2.164 | 6.813 | 13.735 | 23.747 |  [1,1] | SAME |   [128,128,1,1] |  [1,128,56,56] |    [1,1] |  MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D | 15.859 | 2.119 |   2.109 | 6.642 | 20.377 | 24.358 |  [1,1] | SAME |   [512,512,1,1] |  [1,512,14,14] |    [1,1] |  MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D | 23.619 | 2.087 |   2.096 | 6.599 | 26.976 | 24.517 |  [1,1] | SAME |   [512,512,1,1] |  [1,512,14,14] |    [1,1] | MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D | 26.204 | 2.081 |   2.093 | 6.590 | 33.567 | 24.549 |  [1,1] | SAME |   [512,512,1,1] |  [1,512,14,14] |    [1,1] | MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D | 21.038 | 2.036 |   2.091 | 6.585 | 40.152 | 24.569 |  [1,1] | SAME |   [512,512,1,1] |  [1,512,14,14] |    [1,1] |  MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D | 18.465 | 2.034 |   2.082 | 6.554 | 46.706 | 24.684 |  [1,1] | SAME |   [512,512,1,1] |  [1,512,14,14] |    [1,1] |  MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D |  2.709 | 1.984 |   2.058 | 6.482 | 53.188 | 12.480 |  [1,1] | SAME |     [64,32,1,1] | [1,64,112,112] |    [1,1] |  MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D | 12.220 | 1.788 |   1.901 | 5.986 | 59.174 | 27.027 |  [1,1] | SAME |   [256,256,1,1] |  [1,256,28,28] |    [1,1] |  MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6 |
        I statistics.cc:343 |  Conv2D |  0.107 | 1.541 |   1.570 | 4.943 | 64.117 |  6.904 |  [2,2] | SAME |      [32,3,3,3] | [1,32,112,112] |    [1,1] |            MobilenetV1/MobilenetV1/Conv2d_0/Relu6 |
        I statistics.cc:343 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        I statistics.cc:343
        I statistics.cc:343 ----------------------------------------------------------------------------------------------
        I statistics.cc:343                                        Stat by Op Type
        I statistics.cc:343 ----------------------------------------------------------------------------------------------
        I statistics.cc:343 |         Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
        I statistics.cc:343 ----------------------------------------------------------------------------------------------
        I statistics.cc:343 |          Conv2D |    15 |  24.978 | 78.693 |  78.693 | 551,355,392 | 22.074 |           15 |
        I statistics.cc:343 | DepthwiseConv2d |    13 |   6.543 | 20.614 |  99.307 |  17,385,984 |  2.657 |           13 |
        I statistics.cc:343 |       Transpose |     1 |   0.100 |  0.315 |  99.622 |           0 |  0.000 |            1 |
        I statistics.cc:343 |         Pooling |     1 |   0.072 |  0.227 |  99.849 |           0 |  0.000 |            1 |
        I statistics.cc:343 |         Softmax |     1 |   0.041 |  0.129 |  99.978 |           0 |  0.000 |            1 |
        I statistics.cc:343 |         Squeeze |     1 |   0.006 |  0.019 |  99.997 |           0 |  0.000 |            1 |
        I statistics.cc:343 |        Identity |     1 |   0.001 |  0.003 | 100.000 |           0 |  0.000 |            1 |
        I statistics.cc:343 ----------------------------------------------------------------------------------------------
        I statistics.cc:343
        I statistics.cc:343 ---------------------------------------------------------
        I statistics.cc:343           Stat by MACs(Multiply-Accumulation)
        I statistics.cc:343 ---------------------------------------------------------
        I statistics.cc:343 |       total | round | first(G/s) | avg(G/s) |     std |
        I statistics.cc:343 ---------------------------------------------------------
        I statistics.cc:343 | 568,741,376 |   100 |     18.330 |   17.909 | 301.326 |
        I statistics.cc:343 ---------------------------------------------------------
        I statistics.cc:343 ------------------------------------------------------------------------
        I statistics.cc:343                           Summary of Ops' Stat
        I statistics.cc:343 ------------------------------------------------------------------------
        I statistics.cc:343 | round | first(ms) | curr(ms) | min(ms) | max(ms) | avg(ms) |     std |
        I statistics.cc:343 ------------------------------------------------------------------------
        I statistics.cc:343 |   100 |    31.028 |   32.093 |  31.028 |  32.346 |  31.758 | 301.326 |
        I statistics.cc:343 ------------------------------------------------------------------------


===========
Explanation
===========

There are 8 sections of the output information.

1. **Warm Up**

This section lists the time information of warm-up run.
The detailed explanation is list as below.

.. list-table::
    :header-rows: 1

    * - Key
      - Explanation
    * - round
      - the number of round has been run.
    * - first
      - the run time of first round. unit is millisecond.
    * - curr
      - the run time of last round. unit is millisecond.
    * - min
      - the minimal run time of all rounds. unit is millisecond.
    * - max
      - the maximal run time of all rounds. unit is millisecond.
    * - avg
      - the average run time of all rounds. unit is millisecond.
    * - std
      - the standard deviation of all rounds.

2. **Run without statistics**

This section lists the run time information without statistics code.
 the detailed explanation is the same as the section of Warm Up.

3. **Run with statistics**

This section lists the run time information with statistics code,
 the time maybe longer compared with the second section.
 the detailed explanation is the same as the section of Warm Up.

4. **Sort by Run Order**

This section lists the detailed run information of every operator in your model.
The operators is listed based on the run order, Every line is an operator of your model.
The detailed explanation is list as below.

.. list-table::
    :header-rows: 1

    * - Key
      - Explanation
    * - Op Type
      - the type of operator.
    * - Start
      - the start time of the operator. unit is millisecond.
    * - First
      - the run time of first round. unit is millisecond.
    * - Avg
      - the average run time of all rounds. unit is millisecond.
    * - %
      - the percentage of total running time.
    * - cdf%
      - the cumulative percentage of running time.
    * - GMACPS
      - The number of run MACs(multiply-accumulation) per second. the unit is G/s.
    * - Stride
      - the stride parameter of the operator if exist.
    * - Pad
      - the pad parameter of the operator if exist.
    * - Filter Shape
      - the filter shape of the operator if exist.
    * - Output Shape
      - the output shape of the operator.
    * - Dilation
      - the dilation parameter of the operator if exist.
    * - Name
      - the name of the operator.

5. **Sort by Computation time**

This section lists the top-10 most time-consuming operators.
The operators is listed based on the computation time,
the detailed explanation is the same as previous section.

6. **Stat by Op Type**

This section stats the run information about operators based on operator type.

.. list-table::
    :header-rows: 1

    * - Op Type
      - the type of operator.
    * - Count
      - the number of operators with the type.
    * - Avg
      - the average run time of the operator. unit is millisecond.
    * - %
      - the percentage of total running time.
    * - cdf%
      - the cumulative percentage of running time.
    * - MACs
      - The number of MACs(multiply-accumulation).
    * - GMACPS
      - The number of MACs(multiply-accumulation) runs per second. the unit is G/s.
    * - Called times
      - the number of called times in all rounds.

7. **Stat by MACs**

This section stats the MACs information of your model.

.. list-table::
    :header-rows: 1

    * - total
      - the number of MACs of your model.
    * - round
      - the number of round has been run.
    * - First
      - the GMAPS of first round. unit is G/s.
    * - Avg
      - the average GMAPS of all rounds. unit is G/s.
    * - std
      - the standard deviation of all rounds.

8. **Summary of Ops' Stat**

This section lists the run time information which is summation of every operator's run time.
which may be shorter than the model's run time with statistics.
the detailed explanation is the same as the section of Warm Up.
