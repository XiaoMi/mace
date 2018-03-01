#!/bin/bash

Usage() {
  echo "Usage: bash tools/validate_tools.sh model_output_dir generate_data_or_not"
}

if [ $# -lt 2 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

MODEL_OUTPUT_DIR=$1
GENERATE_DATA_OR_NOT=$2

if [ "$GENERATE_DATA_OR_NOT" = 1 ]; then
  rm -rf ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME}
  python tools/validate.py --generate_data true \
     --input_file=${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
      --input_shape="${INPUT_SHAPE}" || exit 1
else
  rm -rf ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME}
  adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} ${MODEL_OUTPUT_DIR}
  python tools/validate.py --model_file ${MODEL_FILE_PATH} \
      --input_file ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
      --mace_out_file ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
      --mace_runtime ${RUNTIME} \
      --input_node ${INPUT_NODE} \
      --output_node ${OUTPUT_NODE} \
      --input_shape ${INPUT_SHAPE} \
      --output_shape ${OUTPUT_SHAPE}
fi
