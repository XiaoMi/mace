#!/bin/bash
set -x
pyenv local 3.6.3
LINES=$(adb devices | grep "\<device\>" | awk '{print $1}')
QCM_ADDRS=()
MTK_ADDRS=()
for ADDR in $LINES
do
  QCM_STR=`adb -s $ADDR shell "grep Hardware /proc/cpuinfo | grep Qualcomm"`
  MTK_STR=`adb -s $ADDR shell "grep Hardware /proc/cpuinfo | egrep MT[0-9]+"`
  if [[ ! -z "$QCM_STR" ]]; then
      QCM_ADDRS+=("$ADDR")
  elif [[ ! -z "$MTK_STR" ]]; then
      MTK_ADDRS+=("$ADDR")
  fi
done

WEEK_OF_DAY=`date | awk '{print $1}'`
NUM_OF_QCM=${#QCM_ADDRS[@]}
NUM_OF_MTK=${#MTK_ADDRS[@]}
# Devices which to be tested on
TEST_ADDRS=()
if [[ "$WEEK_OF_DAY" == "Sat" || "$WEEK_OF_DAY" == "Sun" ]]; then
    # Weekend, test all devices.
    TEST_ADDRS=(${QCM_ADDRS[@]} ${MTK_ADDRS[@]})
else
    # Non-Weekend, test at most 2 devices: 1 Qcom and 1 MTK.
    if [[ "$NUM_OF_QCM" -ge "2" ]]; then
        RANDOM_ARRAY_IDX=$(($RANDOM % $NUM_OF_QCM))
        TEST_ADDRS+=(${QCM_ADDRS[$RANDOM_ARRAY_IDX]})
    else
        TEST_ADDRS+=(${QCM_ADDRS[@]})
    fi

    if [[ "$NUM_OF_MTK" -ge "2" ]]; then
        RANDOM_ARRAY_IDX=$(($RANDOM % $NUM_OF_MTK))
        TEST_ADDRS+=(${MTK_ADDRS[$RANDOM_ARRAY_IDX]})
    else
        TEST_ADDRS+=(${MTK_ADDRS[@]})
    fi
fi

echo "Running for gpu"
for ADDR in ${TEST_ADDRS[@]}; do
  find test/ci-mace-models/ -type f -name "*.yml" | while read YML_FILE_NAME
  do
    RUNTIME_INFO=`grep "runtime:.*gpu" $YML_FILE_NAME`
    if [[ ! -z "$RUNTIME_INFO" ]]; then
      echo $YML_FILE_NAME
      python tools/converter.py convert --config $YML_FILE_NAME || exit 1
      python tools/converter.py run --config $YML_FILE_NAME --validate --devices_to_run $ADDR --target_socs "" || exit 2
    fi
  done
done


echo "Running for cpu"
for ADDR in ${TEST_ADDRS[@]}; do
  find test/ci-mace-models/ -type f -name "*.yml" | while read YML_FILE_NAME
  do
    RUNTIME_INFO=`grep "runtime:.*cpu" $YML_FILE_NAME`
    if [[ ! -z "$RUNTIME_INFO" ]]; then
      echo $YML_FILE_NAME
      python tools/converter.py convert --config $YML_FILE_NAME || exit 1
      python tools/converter.py run --config $YML_FILE_NAME --validate --devices_to_run $ADDR --target_socs "" || exit 2
    fi
  done
done
