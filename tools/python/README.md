# MACE Build and Test Tools

## Clear Workspace
Before you do anything, clear the workspace used by build and test process.
```bash
tools/clear_workspace.sh
```

## Build Engine
Please make sure you have CMake installed.
```bash
RUNTIME=GPU bash tools/cmake/cmake-build-armeabi-v7a.sh
```
which generate libraries in `build/cmake-build/armeabi-v7a`, you can use either static libraries or the `libmace.so` shared library.

You can also build for other target abis. 
The default build command builds engine that runs on CPU, you can modify the cmake file to support other hardware, or you can just set environment variable before building.
```bash
RUNTIME: GPU/HEXAGON/HTA/APU
```

## Model Conversion
When you have prepared your model, the first thing to do is write a model config.

```yaml
models:
  mobilenet_v1:
    platform: tensorflow
    model_file_path: https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb
    model_sha256_checksum: 71b10f540ece33c49a7b51f5d4095fc9bd78ce46ebf0300487b2ee23d71294e6
    subgraphs:
      - input_tensors:
          - input
        input_shapes:
          - 1,224,224,3
        output_tensors:
          - MobilenetV1/Predictions/Reshape_1
        output_shapes:
          - 1,1001
    runtime: gpu

```

The following steps generate output to `build` directory which is the default build and test workspace.
Suppose you have the model config in `../mace-models/mobilenet-v1/mobilenet-v1.yml`. Then run

```bash
python tools/python/convert.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml
```

which generate 4 files in `build/mobilenet_v1/model/`
```
├── mobilenet_v1.pb                (model file)
├── mobilenet_v1.data              (param file)
├── mobilenet_v1_index.html        (visualization page, you can open it in browser)
└── mobilenet_v1.pb_txt            (model text file, which can be for debug use)
```

## Model Test and Benchmark
After model is converted, simply run
```bash
python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --validate
```

Or benchmark the model
```bash
python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --benchmark
```


It will test your model on the device configured in the model config (`runtime`). 
You can also test on other device by specify `--runtime=cpu (dsp/hta/apu)` if you previously build engine for the device.

The log will be shown if `--vlog_level=2` is specified.


## Encrypt Model (optional)
Model can be encrypted by obfuscation.
```bash
python tools/python/encrypt.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml
```
It will override `mobilenet_v1.pb` and `mobilenet_v1.data`. 
If you want to compiled the model into a library, you should use options `--gencode_model --gencode_param` to generate model code, i.e.,

```bash
python tools/python/encrypt.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --gencode_model --gencode_param
```
It will generate model code into `mace/codegen/models` and also generate a helper function `CreateMaceEngineFromCode` in `mace/codegen/engine/mace_engine_factory.h` by which you can create an engine with models built in it.

After that you can rebuild the engine. 
```bash
RUNTIME=GPU RUNMODE=code bash tools/cmake/cmake-build-armeabi-v7a.sh
```
`RUNMODE=code` means you compile and link model library with MACE engine.

When you test the model in code format, you should specify it in the script as follows.
```bash
python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --gencode_model --gencode_param
```
Of course you can generate model code only, and use parameter file.

## Precompile OpenCL (optional)
After you test model on GPU, it will generate compiled OpenCL binary file automatically in `build/mobilenet_v1/opencl` directory.
```bash
└── mobilenet_v1_compiled_opencl_kernel.MIX2S.sdm845.bin
```
It specifies your test platform model and SoC. You can use it in production to accelerate the initialization.


## Auto Tune OpenCL kernels (optional)
MACE can auto tune OpenCL kernels used by models. You can specify `--tune` option.
```bash
python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --tune
```
It will generate OpenCL tuned parameter binary file in `build/mobilenet_v1/opencl` directory.
```bash
└── mobilenet_v1_tuned_opencl_parameter.MIX2S.sdm845.bin
```
It specifies your test platform model and SoC. You can use it in production to reduce latency on GPU.


## Multi Model Support (optional)
If multiple models are configured in config file. After you test it, it will generate more than one tuned parameter files.
Then you need to merge them together.
```bash
python tools/python/gen_opencl.py
```
After that, it will generate one set of files into `build/opencl` directory.

```bash
├── compiled_opencl_kernel.bin
└── tuned_opencl_parameter.bin
```

You can also generate code into the engine by specify `--gencode`, after which you should rebuild the engine.


