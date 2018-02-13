CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

if [ x"$TARGET_ABI" != x"host" ]; then
  adb shell rm -rf $PHONE_DATA_DIR
fi

rm -rf codegen/models codegen/opencl codegen/tuning
