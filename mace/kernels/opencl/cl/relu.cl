__kernel void relu(__global const float *input,
                   __private const int size,
                   __global float *output) {
  int idx = get_global_id(0);

  if (idx + 4 > size) {
    for(; idx < size; ++idx) {
      *(output+idx) = fmax(*(input+idx), 0);
    }
  } else {
    float4 data = vload4(idx, input);
    data = fmax(data, (float4)0);
    vstore4(data, idx, output);
  }
}

__kernel void relux(__global const float *input,
                    __private const float max_limit,
                    __private const int size,
                    __global float *output) {
  int idx = get_global_id(0);

  if (idx + 4 > size) {
    for(; idx < size; ++idx) {
      *(output+idx) = clamp(*(input+idx), 0.0f, max_limit);
    }
  } else {
    float4 data = vload4(idx, input);
    data = clamp(data, (float4)0, (float4)max_limit);
    vstore4(data, idx, output);
  }
}
