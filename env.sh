#!/usr/bin/env bash
LIBMACE_TAG=`git describe --abbrev=0 --tags`

VLOG_LEVEL=0
LIBMACE_SOURCE_DIR=`/bin/pwd`
INPUT_FILE_NAME="model_input"
OUTPUT_FILE_NAME="model.out"
OUTPUT_LIST_FILE="model.list"
TF_MODEL_NAME="model.pb"
TF_MODEL_FILE_PATH=$TF_MODEL_FILE_DIR/$TF_MODEL_NAME
PHONE_DATA_DIR="/data/local/tmp/mace_run"
KERNEL_DIR="${PHONE_DATA_DIR}/cl/"
CODEGEN_DIR=${LIBMACE_SOURCE_DIR}/codegen
MODEL_CODEGEN_DIR=${CODEGEN_DIR}/models/${MODEL_TAG}
CL_CODEGEN_DIR=${CODEGEN_DIR}/opencl
TUNING_CODEGEN_DIR=${CODEGEN_DIR}/tuning
VERSION_SOURCE_PATH=${CODEGEN_DIR}/version

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
GENERATED_MODEL_LIB_PATH="bazel-bin/codegen/${GENERATED_MODEL_LIB_NAME}"
