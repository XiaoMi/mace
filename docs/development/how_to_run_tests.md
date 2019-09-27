How to run tests
=================

To run tests, you need to first cross compile the code, push the binary
into the device and then execute the binary. To automate this process,
MACE provides `tools/bazel_adb_run.py` tool.

You need to make sure your device has been connected to your dev pc before running tests.

Run unit tests
---------------

MACE use [gtest](https://github.com/google/googletest) for unit tests.

* Run all unit tests defined in a Bazel target, for example, run `mace_cc_test`:

For CMake users:

  ```sh
  python tools/python/run_target.py \
      --target_abi=armeabi-v7a --target_socs=all --target_name=mace_cc_test 
  ```

For Bazel users:

  ```sh
  python tools/bazel_adb_run.py --target="//test/ccunit:mace_cc_test" \
                                --run_target=True
  ```

* Run unit tests with [gtest](https://github.com/google/googletest) filter,
for example, run `Conv2dOpTest` unit tests:

For CMake users:

  ```sh
  python tools/python/run_target.py \
      --target_abi=armeabi-v7a --target_socs=all --target_name=mace_cc_test \
      --gtest_filter=Conv2dOpTest*
  ```

For Bazel users:

  ```sh
  python tools/bazel_adb_run.py --target="//test/ccunit:mace_cc_test" \
                                --run_target=True \
                                --args="--gtest_filter=Conv2dOpTest*"
  ```

Run micro benchmarks
--------------------

MACE provides a micro benchmark framework for performance tuning.

* Run all micro benchmarks defined in a Bazel target, for example, run all
`mace_cc_benchmark` micro benchmarks:

For CMake users:

  ```sh
  python tools/python/run_target.py \
      --target_abi=armeabi-v7a --target_socs=all --target_name=mace_cc_benchmark
  ```

For Bazel users:

  ```sh
  python tools/bazel_adb_run.py --target="//test/ccbenchmark:mace_cc_benchmark" \
                                --run_target=True
  ```

* Run micro benchmarks with regex filter, for example, run all `CONV_2D` GPU
micro benchmarks:

For CMake users:

  ```sh
  python tools/python/run_target.py \
      --target_abi=armeabi-v7a --target_socs=all --target_name=mace_cc_benchmark \
      --filter=MACE_BM_CONV_2D_.*_GPU
  ```

For Bazel users:

  ```sh
  python tools/bazel_adb_run.py --target="//test/ccbenchmark:mace_cc_benchmark" \
                                --run_target=True \
                                --args="--filter=MACE_BM_CONV_2D_.*_GPU"
  ```