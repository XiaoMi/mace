Deploy
======

MACE Micro module is written in c++98 and only depends on <cmath>.
We can write a CMake toolchain file to build the program for the special platform.

For Cortex-M MCU
----------------

Now we deploy the MNIST classifier example on a NUCLEO-F767ZI development with the Mbed OS.
Install a GCC Arm Embedded compiler by the terminal.

.. code-block:: sh

    # Installs gcc arm
    sudo apt-get install gcc-arm-none-eabi

Refer to <https://os.mbed.com/docs/mbed-os/v6.3/build-tools/install-and-set-up.html/> to install Mbed OS tools.

Now we can convert the model and build the program,

.. code-block:: sh

    python3 tools/python/convert.py --config=micro/pretrained_models/keras/mnist/mnist-int8.yml --enable_micro
    ./micro/tools/cmake/cmake-build-gcc-arm-none-eabi.sh  -DARM_CPU=cortex-m7 -DMACE_MICRO_ENABLE_CMSIS=ON -DMACE_MICRO_ENABLE_HARDFP=OFF

The "-DARM_CPU=cortex-{m7|m4|..}" is a necessary CMake variable for different series of Arm MCUs.
You can use the Mace Micro install package("build/micro/gcc-arm-none-eabi/install") in yourself project. Here we use "mbed-cli" to compile it

.. code-block:: sh

    # cp the MACE Micro libraries to the workspace directory
    cp build/micro/gcc-arm-none-eabi/install micro/examples/classifier -r
    cd micro/examples/classifier
    # Compile the program
    mbed compile -t GCC_ARM -m NUCLEO_F767ZI -D MICRO_MODEL_NAME=mnist_int8 -D MICRO_DATA_NAME=mnist
    # Flash the program to the development board
    cp BUILD/NUCLEO_F767ZI/GCC_ARM/classifier.bin  /media/$USER/NODE_F767ZI
    # Connet to the default COM port
    sudo chown $USER:$USER  /dev/ttyACM0
    mbed sterm

Press the reset(black) button to run the example again.

For Hexagon DSP
---------------

In the micro/cmake/toolchain folder, there are two hexagon CMake toolchain files for reference, For more details, please goto <https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor/>