#! /bin/bash

python tools/python/convert.py --config micro/pretrained_models/tensorflow/kws/kws-tc_resnet8.yml --enable_micro || exit -1

./micro/tools/cmake/cmake-build-gcc-arm-none-eabi.sh \
-DARM_CPU=cortex-m7 \
-DMACE_MICRO_ENABLE_CMSIS=ON \
-DMACE_MICRO_ENABLE_HARDFP=OFF || exit -1

cp build/micro/gcc-arm-none-eabi/install micro/examples/classifier -r
cp micro/examples/data micro/examples/classifier -r

cd micro/examples/classifier

mbed deploy || exit -1
mbed compile -t GCC_ARM -m NUCLEO_F767ZI -D MICRO_MODEL_NAME=kws_tc_resnet8 -D MICRO_DATA_NAME=kws || exit -1

cd ../../..
