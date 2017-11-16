#include <common.h>

// Supported data type: half/float
__kernel void relu(__global const DATA_TYPE *input,
                   __private const int size,
                   __global DATA_TYPE *output) {
  int idx = get_global_id(0);

  if (idx + 4 > size) {
    for(; idx < size; ++idx) {
      *(output+idx) = fmax(*(input+idx), 0);
    }
  } else {
    VEC_DATA_TYPE(DATA_TYPE,4) data = vload4(idx, input);
    data = fmax(data, (VEC_DATA_TYPE(DATA_TYPE,4))0);
    vstore4(data, idx, output);
  }
}

__kernel void relux(__global const DATA_TYPE *input,
                    __private const DATA_TYPE max_limit,
                    __private const int size,
                    __global DATA_TYPE *output) {
  int idx = get_global_id(0);

  if (idx + 4 > size) {
    for(; idx < size; ++idx) {
      *(output+idx) = clamp(*(input+idx), 0.0f, max_limit);
    }
  } else {
    VEC_DATA_TYPE(DATA_TYPE,4) data = vload4(idx, input);
    data = clamp(data, (VEC_DATA_TYPE(DATA_TYPE,4))0, (VEC_DATA_TYPE(DATA_TYPE,4))max_limit);
    vstore4(data, idx, output);
  }
}
