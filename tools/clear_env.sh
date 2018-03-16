#!/bin/bash

Usage() {
  echo "Usage: bash tools/clear_env.sh target_soc"
}

if [ $# -lt 1 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

TARGET_SOC=$1
DEVICE_ID=`echo_device_id_by_soc $TARGET_SOC`

if [ x"$TARGET_ABI" != x"host" ]; then
  adb -s $DEVICE_ID shell rm -rf $PHONE_DATA_DIR || exit 1
fi

rm -rf mace/codegen/models
git checkout -- mace/codegen/opencl/opencl_compiled_program.cc mace/codegen/tuning/tuning_params.cc
