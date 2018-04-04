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
RESULT_VALUE=`echo_device_id_by_soc $TARGET_SOC`
if [ $? -ne 0 ]; then
  echo $RESULT_VALUE
  exit 1
else
  DEVICE_ID=$RESULT_VALUE
fi

if [ x"$TARGET_ABI" != x"host" ]; then
  adb -s $DEVICE_ID shell rm -rf $PHONE_DATA_DIR || exit 1
fi

rm -rf mace/codegen/models
