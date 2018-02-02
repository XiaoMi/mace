CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

if [ x"$RUNTIME" != x"local" ]; then
  adb shell rm -rf $PHONE_DATA_DIR
  rm -rf codegen/models codegen/opencl codegen/tuning
fi
