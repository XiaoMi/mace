/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef THIRD_PARTY_HTA_HEXAGON_API_H_
#define THIRD_PARTY_HTA_HEXAGON_API_H_

#include "hta_hexagon_nn_ops.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int hexagon_hta_nn_nn_id;

struct input {
	uint32_t src_id;
	uint32_t output_idx;
};

#define NODE_ID_RESERVED_CONSTANT 0

#define MAX_DIMENSIONS 8
struct output {
	uint32_t rank; // dimensions in the tensor
	uint32_t max_sizes[MAX_DIMENSIONS]; // max num elements in each dimension
	uint32_t elementsize; // size of each element
	int32_t zero_offset; // 0 for float / integer values
	float stepsize; // 0 for float/integer values
};

struct perfinfo {
	uint32_t node_id;
	uint32_t executions;
	union {
		uint64_t counter;
		struct {
			uint32_t counter_lo;
			uint32_t counter_hi;
		};
	};
};

typedef struct input hexagon_hta_nn_input;
typedef struct output hexagon_hta_nn_output;
typedef struct perfinfo hexagon_hta_nn_perfinfo;
typedef int32_t hexagon_hta_nn_padding_type;

typedef enum padding_type_enum {
	HTA_NN_PAD_NA = 0,
	HTA_NN_PAD_SAME,
	HTA_NN_PAD_VALID,
	HTA_NN_PAD_MIRROR_REFLECT,
	HTA_NN_PAD_MIRROR_SYMMETRIC,
	HTA_NN_PAD_SAME_CAFFE,
} hta_padding_type;

typedef struct {
	unsigned int batches;
	unsigned int height;
	unsigned int width;
	unsigned int depth;
	unsigned char *data;
	int dataLen;		/* For input and output */
	unsigned int data_valid_len; /* for output only */
	unsigned int unused;
} hexagon_hta_nn_tensordef;

typedef struct hexagon_nn_op_node hexagon_nn_op_node;
struct hexagon_nn_op_node {
  unsigned int node_id;
  hta_op_type operation;
  hta_padding_type padding;
  hexagon_hta_nn_input* inputs;
  int inputsLen;
  hexagon_hta_nn_output* outputs;
  int outputsLen;
};
typedef struct hexagon_nn_const_node hexagon_nn_const_node;
struct hexagon_nn_const_node {
  unsigned int node_id;
  hexagon_hta_nn_tensordef tensor;
};

/* Actual functions in the interface */
/* Returns 0 on success, nonzero on error unless otherwise noted */
/* Configure the hardware and software environment.  Should be called once before doing anything */
int hexagon_hta_nn_config( void );

/* Initialize a new graph, returns a new nn_id or -1 on error */
int hexagon_hta_nn_init(hexagon_hta_nn_nn_id *g);

/* Set debug verbosity.  Default is 0, higher values are more verbose */
int hexagon_hta_nn_set_debug_level(hexagon_hta_nn_nn_id id, int level);

/* Append a node to the graph.  Nodes are executed in the appended order. */
int hexagon_hta_nn_append_node(
	hexagon_hta_nn_nn_id id,
	uint32_t node_id,
	hta_op_type operation,
	hta_padding_type padding,
	const struct input *inputs,
	uint32_t num_inputs,
	const struct output *outputs,
	uint32_t num_outputs);

/*
 * Append a const node into the graph.  The data is copied locally during this
 * call, the caller does not need it to persist.
 */
int hexagon_hta_nn_append_const_node(
	hexagon_hta_nn_nn_id id,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len);

/*
 * Prepare a graph for execution.  Must be done before attempting to execute the graph.
 */
int hexagon_hta_nn_prepare(hexagon_hta_nn_nn_id id);

/* Execute the graph with a single input and a single output. */
int hexagon_hta_nn_execute(
	hexagon_hta_nn_nn_id id,
	uint32_t batches_in,
	uint32_t height_in,
	uint32_t width_in,
	uint32_t depth_in,
	const uint8_t *data_in,
	uint32_t data_len_in,
	uint32_t *batches_out,
	uint32_t *height_out,
	uint32_t *width_out,
	uint32_t *depth_out,
	uint8_t *data_out,
	uint32_t data_out_max,
	uint32_t *data_out_size);

/* Tear down a graph, destroying it and freeing resources.  */
int hexagon_hta_nn_teardown(hexagon_hta_nn_nn_id id);

/* Get the version of the library */
int hexagon_hta_nn_version(int *ver);

/* Execute the graph with a multiple input and a multiple output. */
int hexagon_hta_nn_execute_new(
	hexagon_hta_nn_nn_id id,
	const hexagon_hta_nn_tensordef *inputs,
	uint32_t n_inputs,
	hexagon_hta_nn_tensordef *outputs,
	uint32_t n_outputs);

int hexagon_hta_nn_serialize_size(hexagon_hta_nn_nn_id id, unsigned int *serialized_obj_size_out);
int hexagon_hta_nn_serialize(hexagon_hta_nn_nn_id id, void *buf, unsigned int buf_len);
int hexagon_hta_nn_deserialize(void *buf, unsigned len, hexagon_hta_nn_nn_id *g);

#ifdef __cplusplus
}
#endif

#endif //THIRD_PARTY_HTA_HEXAGON_API_H_
