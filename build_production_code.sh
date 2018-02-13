#!/bin/bash

Usage() {
  echo "Usage: bash tools/build_production_code.sh"
}

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

build_local_target()
{
  BAZEL_TARGET=$1
  bazel build --verbose_failures -c opt --strip always $BAZEL_TARGET \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_OBFUSCATE_LITERALS" \
    --define openmp=true || exit -1
}

build_target()
{
  BAZEL_TARGET=$1
  bazel build --verbose_failures -c opt --strip always $BAZEL_TARGET \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
     --cpu=$ANDROID_ABI \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --define openmp=true \
    --copt="-DMACE_OBFUSCATE_LITERALS" || exit 1
}

if [ x"$RUNTIME" = x"local" ]; then
  build_local_target //codegen:generated_opencl_prod
  build_local_target //codegen:generated_tuning_params
else
  build_target //codegen:generated_opencl_prod
  build_target //codegen:generated_tuning_params
fi
