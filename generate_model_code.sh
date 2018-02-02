#!/bin/bash

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

bazel build //lib/python/tools:tf_converter || exit 1
rm -rf ${MODEL_CODEGEN_DIR}
mkdir -p ${MODEL_CODEGEN_DIR}
if [ ${DSP_MODE} ]; then
    DSP_MODE_FLAG="--dsp_mode=${DSP_MODE}"
fi
bazel-bin/lib/python/tools/tf_converter --input=${TF_MODEL_FILE_PATH} \
                                        --output=${MODEL_CODEGEN_DIR}/model.cc \
                                        --input_node=${TF_INPUT_NODE} \
                                        --output_node=${TF_OUTPUT_NODE} \
                                        --data_type=${DATA_TYPE} \
                                        --runtime=${MACE_RUNTIME} \
                                        --output_type=source \
                                        --template=${LIBMACE_SOURCE_DIR}/lib/python/tools/model.template \
                                        --model_tag=${MODEL_TAG} \
                                        ${DSP_MODE_FLAG} \
                                        --obfuscate=True || exit 1
