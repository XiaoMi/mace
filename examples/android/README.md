Android Demo
=============

How to build
---------------

```sh
cd mace/examples/android
./build.sh dynamic
# if libmace.a is needed, update `macelibrary/CMakeLists.txt` and run with `./build.sh static`.
```

Install
---------------

```sh
# running after build step and in `mace/exampls/android` directory
adb install ./app/build/outputs/apk/app/release/app-app-release.apk
```

Pre-built APK
--------------
Pre-built Android APK can be downloaded [here](https://cnbj1.fds.api.xiaomi.com/mace/demo/mace_android_demo.apk).

Note
--------------
We use two big cores for CPU inference.
