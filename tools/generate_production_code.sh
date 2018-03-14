#!/bin/bash

Usage() {
  echo "Usage: bash tools/generate_production_code.sh cl_bin_dirs pull_or_not"
}

if [ $# -lt 2 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

CL_BIN_DIRS=$1
PULL_OR_NOT=$2

if [ "$PULL_OR_NOT" = 1 ]; then
  CL_BIN_DIR=${CL_BIN_DIRS}
  rm -rf ${CL_BIN_DIR}
  mkdir -p ${CL_BIN_DIR}
  if [ x"$TARGET_ABI" != x"host" ]; then
    adb pull ${KERNEL_DIR}/. ${CL_BIN_DIR} > /dev/null
    adb pull ${PHONE_DATA_DIR}/mace_run.config ${CL_BIN_DIR} > /dev/null
  fi
fi

python mace/python/tools/opencl_codegen.py \
  --cl_binary_dirs=${CL_BIN_DIRS} \
  --output_path=${CL_CODEGEN_DIR}/opencl_compiled_program.cc

python mace/python/tools/binary_codegen.py \
  --binary_dirs=${CL_BIN_DIRS} \
  --binary_file_name=mace_run.config \
  --output_path=${TUNING_CODEGEN_DIR}/tuning_params.cc
