#!/usr/bin/env bash
LIBMACE_TAG=`git describe --abbrev=0 --tags`

MACE_SOURCE_DIR=`/bin/pwd`
INPUT_FILE_NAME="model_input"
OUTPUT_FILE_NAME="model_out"
PHONE_DATA_DIR="/data/local/tmp/mace_run"
KERNEL_DIR="${PHONE_DATA_DIR}/cl/"
CODEGEN_DIR=${MACE_SOURCE_DIR}/mace/codegen
MODEL_CODEGEN_DIR=${CODEGEN_DIR}/models/${MODEL_TAG}
CL_CODEGEN_DIR=${CODEGEN_DIR}/opencl
TUNING_CODEGEN_DIR=${CODEGEN_DIR}/tuning
VERSION_SOURCE_PATH=${CODEGEN_DIR}/version
if [ -z ${EMBED_MODEL_DATA} ]; then
  EMBED_MODEL_DATA=1
fi

if [ x"$RUNTIME" = x"dsp" ]; then
  DATA_TYPE="DT_UINT8"
  DEVICE_TYPE="HEXAGON"
  LIB_FOLDER_NAME="${LIB_FOLDER_NAME}_dsp"
elif [ x"$RUNTIME" = x"gpu" ]; then
  DATA_TYPE="DT_HALF"
  DEVICE_TYPE="OPENCL"
elif [ x"$RUNTIME" = x"cpu" ]; then
  DATA_TYPE="DT_FLOAT"
  DEVICE_TYPE="CPU"
fi

GENERATED_MODEL_LIB_NAME="libgenerated_models.a"
if [ x"$TARGET_ABI" = x"host" ]; then
  GENERATED_MODEL_LIB_NAME="libgenerated_models.pic.a"
fi
GENERATED_MODEL_LIB_PATH="bazel-bin/mace/codegen/${GENERATED_MODEL_LIB_NAME}"
