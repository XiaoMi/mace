Examples
=======

To build the example:
```
bazel build mace/examples:helloworld \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=armeabi-v7a
```
