__kernel void add2(__global const float *input0,
                   __global const float *input1,
                   __private const int size,
                   __global float *output) {
  int idx = get_global_id(0);

  if (idx + 4 > size) {
    for(; idx < size; ++idx) {
      *(output+idx) = *(input0+idx) + *(input1+idx);
    }
  } else {
    float4 in_data0 = vload4(idx, input0);
    float4 in_data1 = vload4(idx, input1);
    vstore4(in_data0+in_data1, idx, output);
  }
}

