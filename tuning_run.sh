#!/bin/bash

Usage() {
  echo "Usage: bash tools/tuning_run.sh model_output_dir round tuning production_mode"
}

if [ $# -lt 4 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

MODEL_OUTPUT_DIR=$1
ROUND=$2
TUNING_OR_NOT=$3
PRODUCTION_MODE=$4
RESTART_ROUND=$5

if [ x"$TARGET_ABI" = x"host" ]; then
  MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
  ${MODEL_OUTPUT_DIR}/mace_run \
    --input_node="${INPUT_NODES}" \
    --input_shape="${INPUT_SHAPES}"\
    --output_node="${OUTPUT_NODES}" \
    --output_shape="${OUTPUT_SHAPES}"\
    --input_file=${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
    --output_file=${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
    --model_data_file=${MODEL_OUTPUT_DIR}/${MODEL_TAG}.data \
    --device=${DEVICE_TYPE}   \
    --round=1 \
    --restart_round=1 || exit 1
else
  if [[ "${TUNING_OR_NOT}" != "0" && "$PRODUCTION_MODE" != 1 ]];then
    tuning_flag=1
  else
    tuning_flag=0
  fi
  
  adb shell "mkdir -p ${PHONE_DATA_DIR}" || exit 1
  if [ "$PRODUCTION_MODE" = 0 ]; then
    adb shell "mkdir -p ${KERNEL_DIR}" || exit 1
  fi

  IFS=',' read -r -a INPUT_NAMES <<< "${INPUT_NODES}"
  for NAME in "${INPUT_NAMES[@]}";do
    FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
    adb push ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME}_${FORMATTED_NAME} ${PHONE_DATA_DIR} || exit 1
  done

  adb push ${MODEL_OUTPUT_DIR}/mace_run ${PHONE_DATA_DIR} || exit 1
  if [ "$EMBED_MODEL_DATA" = 0 ]; then
    adb push ${MODEL_OUTPUT_DIR}/${MODEL_TAG}.data ${PHONE_DATA_DIR} || exit 1
  fi
  adb push lib/hexagon/libhexagon_controller.so ${PHONE_DATA_DIR} || exit 1
  
  mace_adb_output=`adb </dev/null shell \
    "LD_LIBRARY_PATH=${PHONE_DATA_DIR} \
    MACE_TUNING=${tuning_flag} \
    MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
    MACE_RUN_PARAMETER_PATH=${PHONE_DATA_DIR}/mace_run.config \
    MACE_KERNEL_PATH=$KERNEL_DIR \
    MACE_LIMIT_OPENCL_KERNEL_TIME=${LIMIT_OPENCL_KERNEL_TIME} \
    ${PHONE_DATA_DIR}/mace_run \
    --input_node="${INPUT_NODES}" \
    --input_shape="${INPUT_SHAPES}"\
    --output_node="${OUTPUT_NODES}" \
    --output_shape="${OUTPUT_SHAPES}"\
    --input_file=${PHONE_DATA_DIR}/${INPUT_FILE_NAME} \
    --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
    --model_data_file=${PHONE_DATA_DIR}/${MODEL_TAG}.data \
    --device=${DEVICE_TYPE}   \
    --round=$ROUND \
    --restart_round=$RESTART_ROUND; echo \\$?"` || exit 1
  echo "$mace_adb_output" | head -n -1

  mace_adb_return_code=`echo "$mace_adb_output" | tail -1`
  if [ $mace_adb_return_code -ne 0 ]; then
    exit 1
  fi
fi
