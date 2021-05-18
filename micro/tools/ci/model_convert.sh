#! /bin/bash

set +x
CONF_FILE=micro/pretrained_models/har-cnn/har-cnn.yml
python tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
python tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name har_cnn || exit -1
python tools/python/run_micro.py --config $CONF_FILE --model_name har_cnn --build --benchmark || exit -1

git clean -xdf micro/codegen

CONF_FILE=micro/pretrained_models/har-cnn/har-cnn-bf16.yml
python tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
python tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name har_cnn || exit -1

git clean -xdf micro/codegen

CONF_FILE=micro/pretrained_models/keras/mnist/mnist.yml
python3 tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
python3 tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name mnist || exit -1

git clean -xdf micro/codegen

CONF_FILE=micro/pretrained_models/keras/mnist/mnist-int8.yml
python3 tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
python3 tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name mnist_int8 || exit -1

git clean -xdf micro/codegen

CONF_FILE=micro/pretrained_models/keras/har/har.yml
python3 tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
python3 tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name har || exit -1

git clean -xdf micro/codegen

# CONF_FILE=micro/pretrained_models/keras/har/har-int8.yml
# python3 tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
# python3 tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name har_int8 || exit -1

git clean -xdf micro/codegen

CONF_FILE=micro/pretrained_models/tensorflow/kws/kws-tc_resnet8.yml
python tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
python tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name kws_tc_resnet8 || exit -1

git clean -xdf micro/codegen

CONF_FILE=micro/pretrained_models/tensorflow/kws/kws-tc_resnet8-bf16.yml
python tools/python/convert.py --config=${CONF_FILE} --enable_micro || exit -1
python tools/python/run_micro.py --config $CONF_FILE --build --validate --model_name kws_tc_resnet8_bf16 || exit -1
