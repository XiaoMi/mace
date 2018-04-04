#!/usr/bin/env bash

Usage() {
  echo "Usage: bash tools/genenrate_opencl_code.sh type [target_soc] [cl_bin_dirs] [pull_or_not]"
}

if [ $# -lt 1 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

TYPE=$1
TARGET_SOC=$2
CL_BIN_DIRS=$3
PULL_OR_NOT=$4

mkdir -p ${CL_CODEGEN_DIR}

if [ x"$TYPE" == x"source" ];then
  python mace/python/tools/encrypt_opencl_codegen.py \
    --cl_kernel_dir=./mace/kernels/opencl/cl/ \
    --output_path=${CODEGEN_DIR}/opencl/opencl_encrypt_program.cc || exit 1
elif [ x"$#" == x"1" ];then

  python mace/python/tools/opencl_codegen.py \
    --output_path=${CL_CODEGEN_DIR}/opencl_compiled_program.cc || exit 1

else
  RESULT_VALUE=`echo_device_id_by_soc $TARGET_SOC`
  if [ $? -ne 0 ]; then
    echo $RESULT_VALUE
    exit 1
  else
    DEVICE_ID=$RESULT_VALUE
  fi

  if [ "$PULL_OR_NOT" = 1 ]; then
    CL_BIN_DIR=${CL_BIN_DIRS}
    rm -rf ${CL_BIN_DIR}
    mkdir -p ${CL_BIN_DIR}
    if [ x"$TARGET_ABI" != x"host" ]; then
      adb -s $DEVICE_ID pull ${COMPILED_PROGRAM_DIR}/. ${CL_BIN_DIR} > /dev/null
    fi
  fi

  python mace/python/tools/opencl_codegen.py \
    --cl_binary_dirs=${CL_BIN_DIRS} \
    --output_path=${CL_CODEGEN_DIR}/opencl_compiled_program.cc || exit 1
fi

