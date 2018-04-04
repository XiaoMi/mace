#!/bin/bash

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

mkdir -p ${TUNING_CODEGEN_DIR}

if [ "$#" -eq "0" ]; then
  python mace/python/tools/binary_codegen.py \
    --binary_file_name=mace_run.config \
    --output_path=${TUNING_CODEGEN_DIR}/tuning_params.cc
else

  TARGET_SOC=$1
  BIN_DIRS=$2
  PULL_OR_NOT=$3

  RESULT_VALUE=`echo_device_id_by_soc $TARGET_SOC`
  if [ $? -ne 0 ]; then
    echo $RESULT_VALUE
    exit 1
  else
    DEVICE_ID=$RESULT_VALUE
  fi

  if [ "$PULL_OR_NOT" = 1 ]; then
    rm -rf ${BIN_DIRS}
    mkdir -p ${BIN_DIRS}
    if [ x"$TARGET_ABI" != x"host" ]; then
      adb -s $DEVICE_ID pull ${PHONE_DATA_DIR}/mace_run.config ${BIN_DIRS} > /dev/null
    fi
  fi

  python mace/python/tools/binary_codegen.py \
  --binary_dirs=${BIN_DIRS} \
  --binary_file_name=mace_run.config \
  --output_path=${TUNING_CODEGEN_DIR}/tuning_params.cc
fi


