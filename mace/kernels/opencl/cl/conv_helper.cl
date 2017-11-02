float4 conv1x3_s1(const float *input_ptr,
                  const float *filter_ptr) {
  float4 row0 = vload4(0, input_ptr);
  float2 input1 = vload2(0, input_ptr+4);
  float4 row1 = (float4)(row0.s123, input1.s0);
  float4 row2 = (float4)(row0.s23, input1.s01);
  float3 filter_values = vload3(0, filter_ptr);
  return (float4)filter_values.s0 * row0 +
         (float4)filter_values.s1 * row1 +
         (float4)filter_values.s2 * row2;
}

float4 conv1x3_s2(const float *input_ptr,
                  const float *filter_ptr) {
  float8 input = vload8(0, input_ptr);
  float4 row0 = input.even;
  float4 row1 = input.odd;
  float4 row2 = (float4)(row0.s123, input_ptr[8]);
  float3 filter_values = vload3(0, filter_ptr);
  return (float4)filter_values.s0 * row0 +
         (float4)filter_values.s1 * row1 +
         (float4)filter_values.s2 * row2;
}

float conv3x3(const float *input_ptr,
              const float *filter_ptr,
              const int row_width) {
  float3 input_value = vload3(0, input_ptr);
  float3 filter_value = vload3(0, filter_ptr);
  float3 res = input_value * filter_value;
  input_ptr += row_width;
  input_value = vload3(0, input_ptr);
  filter_value = vload3(1, filter_ptr);
  res += input_value * filter_value;
  input_ptr += row_width;
  input_value = vload3(0, input_ptr);
  filter_value = vload3(2, filter_ptr);
  res += input_value * filter_value;

  return res.s0 + res.s1 + res.s2;
}
