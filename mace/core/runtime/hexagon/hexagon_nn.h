#ifndef _HEXAGON_NN_H
#define _HEXAGON_NN_H
#ifndef __QAIC_HEADER
#define __QAIC_HEADER(ff) ff
#endif //__QAIC_HEADER

#ifndef __QAIC_HEADER_EXPORT
#define __QAIC_HEADER_EXPORT
#endif // __QAIC_HEADER_EXPORT

#ifndef __QAIC_HEADER_ATTRIBUTE
#define __QAIC_HEADER_ATTRIBUTE
#endif // __QAIC_HEADER_ATTRIBUTE

#ifndef __QAIC_IMPL
#define __QAIC_IMPL(ff) ff
#endif //__QAIC_IMPL

#ifndef __QAIC_IMPL_EXPORT
#define __QAIC_IMPL_EXPORT
#endif // __QAIC_IMPL_EXPORT

#ifndef __QAIC_IMPL_ATTRIBUTE
#define __QAIC_IMPL_ATTRIBUTE
#endif // __QAIC_IMPL_ATTRIBUTE
#ifdef __cplusplus
extern "C" {
#endif
#if !defined(__QAIC_STRING1_OBJECT_DEFINED__) && !defined(__STRING1_OBJECT__)
#define __QAIC_STRING1_OBJECT_DEFINED__
#define __STRING1_OBJECT__
typedef struct _cstring1_s {
  char* data;
  int dataLen;
} _cstring1_t;

#endif /* __QAIC_STRING1_OBJECT_DEFINED__ */
typedef struct hexagon_nn_input hexagon_nn_input;
struct hexagon_nn_input {
  unsigned int src_id;
  unsigned int output_idx;
};
typedef struct hexagon_nn_output hexagon_nn_output;
struct hexagon_nn_output {
  unsigned int max_size;
  unsigned int unused;
};
typedef struct hexagon_nn_perfinfo hexagon_nn_perfinfo;
struct hexagon_nn_perfinfo {
  unsigned int node_id;
  unsigned int node_type;
  unsigned int executions;
  unsigned int unused;
  unsigned int counter_lo;
  unsigned int counter_hi;
};
typedef int hexagon_nn_nn_id;
enum hexagon_nn_padding_type {
  NN_PAD_NA,
  NN_PAD_SAME,
  NN_PAD_VALID,
  NN_PAD_MIRROR_REFLECT,
  NN_PAD_MIRROR_SYMMETRIC,
  NN_PAD_SAME_CAFFE,
  _32BIT_PLACEHOLDER_hexagon_nn_padding_type = 0x7fffffff
};
typedef enum hexagon_nn_padding_type hexagon_nn_padding_type;
typedef struct hexagon_nn_tensordef hexagon_nn_tensordef;
struct hexagon_nn_tensordef {
  unsigned int batches;
  unsigned int height;
  unsigned int width;
  unsigned int depth;
  unsigned char* data;
  int dataLen;
  unsigned int data_valid_len;
  unsigned int unused;
};
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_config)(void) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_init)(void) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_debug_level)(hexagon_nn_nn_id id, int level) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_snpprint)(hexagon_nn_nn_id id, unsigned char* buf, int bufLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_getlog)(hexagon_nn_nn_id id, unsigned char* buf, int bufLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_node)(hexagon_nn_nn_id id, unsigned int node_id, unsigned int operation, hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen, const hexagon_nn_output* outputs, int outputsLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_const_node)(hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_prepare)(hexagon_nn_nn_id id) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_execute)(hexagon_nn_nn_id id, unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in, const unsigned char* data_in, int data_inLen, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_teardown)(hexagon_nn_nn_id id) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_powersave_level)(unsigned int level) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_perfinfo)(hexagon_nn_nn_id id, hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_reset_perfinfo)(hexagon_nn_nn_id id, unsigned int event) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_last_execution_cycles)(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_version)(int* ver) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_op_name_to_id)(const char* name, unsigned int* node_id) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_op_id_to_name)(unsigned int node_id, char* name, int nameLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_disable_dcvs)(void) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_GetHexagonBinaryVersion)(int* ver) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_PrintLog)(const unsigned char* buf, int bufLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_execute_new)(hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen) __QAIC_HEADER_ATTRIBUTE;
#ifdef __cplusplus
}
#endif
#endif //_HEXAGON_NN_H
