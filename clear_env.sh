CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

adb shell rm -rf $PHONE_DATA_DIR
rm -rf codegen/models codegen/opencl codegen/tuning
