#!/bin/bash

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

bazel build //lib/python/tools:converter || exit 1
rm -rf ${MODEL_CODEGEN_DIR}
mkdir -p ${MODEL_CODEGEN_DIR}
if [ ${DSP_MODE} ]; then
    DSP_MODE_FLAG="--dsp_mode=${DSP_MODE}"
fi

OBFUSCATE=True
if [ "${BENCHMARK_FLAG}" = "1" ]; then
  OBFUSCATE=False
fi

bazel-bin/lib/python/tools/tf_converter --platform=${PLATFORM} \
                                        --model_file=${MODEL_FILE_PATH} \
                                        --weight_file=${WEIGHT_FILE_PATH} \
                                        --model_checksum=${MODEL_SHA256_CHECKSUM} \
                                        --output=${MODEL_CODEGEN_DIR}/model.cc \
                                        --input_node=${INPUT_NODE} \
                                        --output_node=${OUTPUT_NODE} \
                                        --data_type=${DATA_TYPE} \
                                        --runtime=${RUNTIME} \
                                        --output_type=source \
                                        --template=${LIBMACE_SOURCE_DIR}/lib/python/tools/model.template \
                                        --model_tag=${MODEL_TAG} \
                                        --input_shape=${INPUT_SHAPE} \
                                        ${DSP_MODE_FLAG} \
                                        --embed_model_data=${EMBED_MODEL_DATA} \
                                        --obfuscate=${OBFUSCATE} || exit 1
