#!/usr/bin/env bash
LIBMACE_TAG=`git describe --abbrev=0 --tags`

MACE_SOURCE_DIR=`/bin/pwd`
PHONE_DATA_DIR="/data/local/tmp/mace_run"
COMPILED_PROGRAM_DIR="${PHONE_DATA_DIR}/cl_program/"
CODEGEN_DIR=${MACE_SOURCE_DIR}/mace/codegen
MODEL_CODEGEN_DIR=${CODEGEN_DIR}/models/${MODEL_TAG}
CL_CODEGEN_DIR=${CODEGEN_DIR}/opencl
TUNING_CODEGEN_DIR=${CODEGEN_DIR}/tuning
VERSION_SOURCE_PATH=${CODEGEN_DIR}/version
CL_BUILT_KERNEL_FILE_NAME=mace_cl_compiled_program.bin
CL_PLATFORM_INFO_FILE_NAME=mace_cl_platform_info.txt
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
elif [ x"$RUNTIME" = x"neon" ]; then
  DATA_TYPE="DT_FLOAT"
  DEVICE_TYPE="NEON"
fi

GENERATED_MODEL_LIB_NAME="libgenerated_models.a"
if [ x"$TARGET_ABI" = x"host" ]; then
  GENERATED_MODEL_LIB_NAME="libgenerated_models.pic.a"
fi
GENERATED_MODEL_LIB_PATH="bazel-bin/mace/codegen/${GENERATED_MODEL_LIB_NAME}"

echo_device_id_by_soc()
{
  TARGET_SOC=$1
  for device in `adb devices | grep "^[A-Za-z0-9]\+[[:space:]]\+device$"| cut -f1`; do
    device_soc=`adb -s ${device} shell getprop | grep ro.board.platform | cut -d [ -f3 | cut -d ] -f1`
    if [ x"$TARGET_SOC" = x"$device_soc" ]; then
      echo "$device"
      return 0
    fi
  done

  echo "MACE ERROR: Not found device with soc ${TARGET_SOC}"
  return 1
}
