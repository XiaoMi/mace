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

if [ x"$TARGET_ABI" = x"host" ]; then
  MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
  ${MODEL_OUTPUT_DIR}/mace_run \
      --input_shape="${INPUT_SHAPE}"\
      --output_shape="${OUTPUT_SHAPE}"\
      --input_file=${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
      --output_file=${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
      --model_data_file=${MODEL_OUTPUT_DIR}/${MODEL_TAG}.data \
      --device=${DEVICE_TYPE}   \
      --round=1 || exit 1
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
  adb push ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} ${PHONE_DATA_DIR} || exit 1
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
    --input_shape=${INPUT_SHAPE}\
    --output_shape=${OUTPUT_SHAPE}\
    --input_file=${PHONE_DATA_DIR}/${INPUT_FILE_NAME} \
    --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
    --model_data_file=${PHONE_DATA_DIR}/${MODEL_TAG}.data \
    --device=${DEVICE_TYPE}   \
    --round=$ROUND; echo \\$?"` || exit 1
  echo "$mace_adb_output" | head -n -1

  mace_adb_return_code=`echo "$mace_adb_output" | tail -1`
  if [ $mace_adb_return_code -ne 0 ]; then
    exit 1
  fi
fi
