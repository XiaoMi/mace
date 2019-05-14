// Copyright 2019 MediaTek Inc. All rights reserved.

#pragma once

enum apu_act_mode {
    APU_ACT_NONE = 0,
    APU_ACT_RELU = 1,
    APU_ACT_RELU6 = 2,
};

enum apu_pooling_mode {
    APU_POOLING_UNDEFINED = 0,
    APU_POOLING_AVG = 1,
    APU_POOLING_MAX = 2,
};

enum apu_eltwise_mode {
    APU_ELTWISE_UNDEFINED = 0,
    APU_ELTWISE_ADD = 1,
    APU_ELTWISE_SUB = 2,
    APU_ELTWISE_MUL = 3,
    APU_ELTWISE_MIN = 4,
    APU_ELTWISE_MAX = 5,
};

enum apu_data_type {
    APU_DATA_TYPE_UNDEFINED = 0,
    APU_DATA_TYPE_FLOAT = 1,
    APU_DATA_TYPE_UINT8 = 2,
    APU_DATA_TYPE_HALF = 3,
    APU_DATA_TYPE_INT32 = 4,
};

enum apu_tensor_type {
    APU_TENSOR_UNDEFINED = 0,
    APU_TENSOR_CONST_DATA = 1,
    APU_TENSOR_CONST_ARGUMENT = 2,
    APU_TENSOR_MODEL_INPUT = 3,
    APU_TENSOR_OP_OUTPUT = 4,
};

#define APU_TENSOR_MAX_DIMS 4

struct apu_tensor {
    int tensor_id;
    apu_tensor_type tensor_type;
    apu_data_type data_type;
    float scale;
    int zero_point;
    int dims[APU_TENSOR_MAX_DIMS];
    int dim_size;
    void* data_buf;
};

#define APU_OP_TYPE_MAX_SIZE 32

struct apu_operator {
    char type[APU_OP_TYPE_MAX_SIZE];
    int* input_ids;
    int input_size;
    apu_tensor output;
    int op_mode;  // for pooling and eltwise
    apu_act_mode act_mode;
};

class ApuFrontend {
 public:
    ApuFrontend();
    ~ApuFrontend();

    bool InitGraph(int const_tensor_size, const apu_tensor* const_tensors,
                   int input_tensor_size, const apu_tensor* input_tensors,
                   int output_tensor_size, const int* output_tensor_ids,
                   void** output_buffers,
                   int operator_size, const apu_operator* operators,
                   bool print_model);
    bool RunGraph();
    bool UninitGraph();

 private:
    class Impl;
    ApuFrontend::Impl* impl;
};
