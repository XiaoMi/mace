Examples
=======

* Build the example (e.g., with armeabi-v7a target)
```
# To enable debug mode build, use '-c dbg' flag.
# To check the underlying commands executed, use '-s' flag.
# TO check the failed command, use '--verbose_failures' flag.

bazel build -c opt mace/examples:helloworld \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=armeabi-v7a
```

* To run adb inside docker, the container network should use 'host'
```
docker run -it --net=host mace-dev /bin/bash
```

* Push and run the example
```
adb shell "mkdir /data/local/tmp/helloword"
adb shell push bazel-bin/mace/examples/helloworld /data/local/tmp/helloword
adb shell /data/local/tmp/helloword/helloworld
```

* Check the logs
```
adb logcat | grep MACE
```
