How to debug
==============

Log debug info
--------------------------
Mace defines two sorts of logs: one is for users (LOG), the other is for developers (VLOG).

LOG includes four levels, i.e, ``INFO``, ``WARNING``, ``ERROR``, ``FATAL``;
Environment variable ``MACE_CPP_MIN_LOG_LEVEL`` can be set to specify log level of users, e.g.,
``set MACE_CPP_MIN_LOG_LEVEL=0`` will enable ``INFO`` log level, while ``set MACE_CPP_MIN_LOG_LEVEL=4`` will enable ``FATAL`` log level.


VLOG level is specified by numbers, e.g., 0, 1, 2. Environment variable ``MACE_CPP_MIN_VLOG_LEVEL`` can be set to specify vlog level.
Logs with higher levels than which is specified will be printed. So simply specifying a very large level number will make all logs printed.

By using Mace run tool, vlog level can be easily set by option, e.g.,

	.. code:: sh

		python tools/converter.py run --config /path/to/model.yml --vlog_level=2


If models are run on android, you might need to use ``adb logcat`` to view logs.


Debug memory usage
--------------------------
The simplest way to debug process memory usage is to use ``top`` command. With ``-H`` option, it can also show thread info.
For android, if you need more memory info, e.g., memory used of all categories, ``adb shell dumpsys meminfo`` will help.
By watching memory usage, you can check if memory usage meets expectations or if any leak happens.


Debug using GDB
--------------------------
GDB can be used as the last resort, as it is powerful that it can trace stacks of your process. If you run models on android,
things may be a little bit complicated.

	.. code:: sh

		# push gdbserver to your phone
		adb push $ANDROID_NDK_HOME/prebuilt/android-arm64/gdbserver/gdbserver /data/local/tmp/


		# set system env, pull system libs and bins to host
		export SYSTEM_LIB=/path/to/android/system_lib
		export SYSTEM_BIN=/path/to/android/system_bin
		mkdir -p $SYSTEM_LIB
		adb pull /system/lib/. $SYSTEM_LIB
		mkdir -p $SYSTEM_BIN
		adb pull /system/bin/. $SYSTEM_BIN


		# Suppose ndk compiler used to compile Mace is of android-21
		export PLATFORMS_21_LIB=$ANDROID_NDK_HOME/platforms/android-21/arch-arm/usr/lib/


		# start gdbserver，make gdb listen to port 6000
		# adb shell /data/local/tmp/gdbserver :6000 /path/to/binary/on/phone/example_bin
		adb shell LD_LIBRARY_PATH=/dir/to/dynamic/library/on/phone/ /data/local/tmp/gdbserver :6000 /data/local/tmp/mace_run/example_bin
		# or attach a running process
		adb shell /data/local/tmp/gdbserver :6000 --attach 8700
		# forward tcp port
		adb forward tcp:6000 tcp:6000


		# use gdb on host to execute binary
		$ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/gdb [/path/to/binary/on/host/example_bin]


		# connect remote port after starting gdb command
		target remote :6000


		# set lib path
		set solib-search-path $SYSTEM_LIB:$SYSTEM_BIN:$PLATFORMS_21_LIB

		# then you can use it as host gdb, e.g.,
		bt







