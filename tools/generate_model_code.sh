#!/bin/bash

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

bazel build //mace/python/tools:converter || exit 1
rm -rf ${MODEL_CODEGEN_DIR}
mkdir -p ${MODEL_CODEGEN_DIR}
if [ ${DSP_MODE} ]; then
  DSP_MODE_FLAG="--dsp_mode=${DSP_MODE}"
fi

PYTHONUNBUFFERED=1 bazel-bin/mace/python/tools/converter --platform=${PLATFORM} \
                                     --model_file=${MODEL_FILE_PATH} \
                                     --weight_file=${WEIGHT_FILE_PATH} \
                                     --model_checksum=${MODEL_SHA256_CHECKSUM} \
                                     --output=${MODEL_CODEGEN_DIR}/model.cc \
                                     --input_node=${INPUT_NODES} \
                                     --output_node=${OUTPUT_NODES} \
                                     --data_type=${DATA_TYPE} \
                                     --runtime=${RUNTIME} \
                                     --output_type=source \
                                     --template=${MACE_SOURCE_DIR}/mace/python/tools \
                                     --model_tag=${MODEL_TAG} \
                                     --input_shape=${INPUT_SHAPES} \
                                     ${DSP_MODE_FLAG} \
                                     --embed_model_data=${EMBED_MODEL_DATA} \
                                     --winograd=${FAST_CONV} \
                                     --obfuscate=${OBFUSCATE} || exit 1
