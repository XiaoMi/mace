// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/core/runtime/hexagon/hexagon_controller.h"
#include "mace/core/runtime/hexagon/hexagon_nn.h"

int hexagon_controller_InitHexagonWithMaxAttributes(int enable_dcvs,
                                                    int bus_usage) {
  return 0;
}

int hexagon_controller_DeInitHexagon() { return 0; }

__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_config)(void)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_init)(void)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_debug_level)(
    hexagon_nn_nn_id id, int level) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_graph_mode)(
    hexagon_nn_nn_id id, int mode) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_snpprint)(hexagon_nn_nn_id id,
                                                            unsigned char *buf,
                                                            int bufLen)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_getlog)(hexagon_nn_nn_id id,
                                                          unsigned char *buf,
                                                          int bufLen)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_node)(
    hexagon_nn_nn_id id,
    unsigned int node_id,
    unsigned int operation,
    hexagon_nn_padding_type padding,
    const hexagon_nn_input *inputs,
    int inputsLen,
    const hexagon_nn_output *outputs,
    int outputsLen) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_node_list)(
    hexagon_nn_nn_id id,
    const hexagon_nn_op_node *ops,
    int opsLen) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_const_node)(
    hexagon_nn_nn_id id,
    unsigned int node_id,
    unsigned int batches,
    unsigned int height,
    unsigned int width,
    unsigned int depth,
    const unsigned char *data,
    int dataLen) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_const_node_list)(
    hexagon_nn_nn_id id,
    const hexagon_nn_const_node *consts,
    int constsLen) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_prepare)(hexagon_nn_nn_id id)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_execute)(
    hexagon_nn_nn_id id,
    unsigned int batches_in,
    unsigned int height_in,
    unsigned int width_in,
    unsigned int depth_in,
    const unsigned char *data_in,
    int data_inLen,
    unsigned int *batches_out,
    unsigned int *height_out,
    unsigned int *width_out,
    unsigned int *depth_out,
    unsigned char *data_out,
    int data_outLen,
    unsigned int *data_len_out) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_teardown)(hexagon_nn_nn_id id)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_powersave_level)(
    unsigned int level) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_perfinfo)(
    hexagon_nn_nn_id id,
    hexagon_nn_perfinfo *info_out,
    int info_outLen,
    unsigned int *n_items) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_reset_perfinfo)(
    hexagon_nn_nn_id id, unsigned int event) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_last_execution_cycles)(
    hexagon_nn_nn_id id,
    unsigned int *cycles_lo,
    unsigned int *cycles_hi) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_version)(int *ver)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_op_name_to_id)(
    const char *name, unsigned int *node_id) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_op_id_to_name)(
    unsigned int node_id, char *name, int nameLen) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_disable_dcvs)(void)
    __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_GetHexagonBinaryVersion)(
    int *ver) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_PrintLog)(
    const unsigned char *buf, int bufLen) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_execute_new)(
    hexagon_nn_nn_id id,
    const hexagon_nn_tensordef *inputs,
    int inputsLen,
    hexagon_nn_tensordef *outputs,
    int outputsLen) __QAIC_HEADER_ATTRIBUTE {
  return 0;
}
