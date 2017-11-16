#include <common.h>

// Supported data type: half/float
__kernel void add2(__global const DATA_TYPE *input0,
                   __global const DATA_TYPE *input1,
                   __private const int size,
                   __global DATA_TYPE *output) {
  int idx = get_global_id(0);

  if (idx + 4 > size) {
    for(; idx < size; ++idx) {
      *(output+idx) = *(input0+idx) + *(input1+idx);
    }
  } else {
    VEC_DATA_TYPE(DATA_TYPE,4) in_data0 = vload4(idx, input0);
    VEC_DATA_TYPE(DATA_TYPE,4) in_data1 = vload4(idx, input1);
    vstore4(in_data0+in_data1, idx, output);
  }
}

