CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

if [ x"$TARGET_ABI" != x"host" ]; then
  adb shell rm -rf $PHONE_DATA_DIR
fi

rm -rf mace/codegen/models
git checkout -- mace/codegen/opencl/opencl_compiled_program.cc mace/codegen/tuning/tuning_params.cc
