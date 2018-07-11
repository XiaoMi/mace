How to run tests
=================

To run tests, you need to first cross compile the code, push the binary
into the device and then execute the binary. To automate this process,
MACE provides `tools/bazel_adb_run.py` tool.

You need to make sure your device has been connected to your dev pc before running tests.

Run unit tests
---------------

MACE use [gtest](https://github.com/google/googletest) for unit tests.

* Run all unit tests defined in a Bazel target, for example, run `ops_test`:

  ```sh
  python tools/bazel_adb_run.py --target="//mace/ops:ops_test" \
                                --run_target=True
  ```

* Run unit tests with [gtest](https://github.com/google/googletest) filter,
for example, run `Conv2dOpTest` unit tests:

  ```sh
  python tools/bazel_adb_run.py --target="//mace/ops:ops_test" \
                                --run_target=True \
                                --args="--gtest_filter=Conv2dOpTest*"
  ```

Run micro benchmarks
--------------------

MACE provides a micro benchmark framework for performance tuning.

* Run all micro benchmarks defined in a Bazel target, for example, run all
`ops_benchmark` micro benchmarks:

  ```sh
  python tools/bazel_adb_run.py --target="//mace/ops:ops_benchmark" \
                                --run_target=True
  ```

* Run micro benchmarks with regex filter, for example, run all `CONV_2D` GPU
micro benchmarks:

  ```sh
  python tools/bazel_adb_run.py --target="//mace/ops:ops_benchmark" \
                                --run_target=True \
                                --args="--filter=MACE_BM_CONV_2D_.*_GPU"
  ```
