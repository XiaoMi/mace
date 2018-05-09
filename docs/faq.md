Frequently Asked Questions
==========================

Why is the generated static library file size so huge?
-------------------------------------------------------
The static library is simply an archive of a set of object files which are
intermediate and contains many extra information, please check whether the
final binary file size is as expected.

Why is the generated binary file (including shared library) size so huge?
-------------------------------------------------------------------------
When compiling the model into C++ code, the final binary may contains extra
debug symbols, they usually takes a lot of space. Try to strip the shared
library or binary. The common overhead of the file size including the compiled
model (excluding the model weights) after the strip should be less than 2MB.
If the model weights is embedded into the binary, the extra overhead should be
around {model weights size in float32}/2.

OpenCL allocator failed with CL_OUT_OF_RESOURCES
------------------------------------------------
OpenCL runtime usually requires continuous virtual memory for its image buffer,
the error will occur when the OpenCL driver can't find the continuous space
due to high memory usage or fragmentation. Several solutions can be tried:

* Change the model by reducing its memory usage
* Split the Op with the biggest single memory buffer
* Changed from armeabi-v7a to arm64-v8a to expand the virtual address space
* Reduce the memory consumption of other modules of the same process

Why the performance is worce than the official result for the same model?
-------------------------------------------------------------------------
The power options may not set properly, see `mace/public/mace_runtime.h` for
details.

Why the UI is getting poor responsiveness when running model with GPU runtime?
------------------------------------------------------------------------------
Try to set `limit_opencl_kernel_time` to `1`. If still not resolved, try to
modify the source code to use even smaller time intervals or changed to CPU
or DSP runtime.

How to include more than one deployment files in one application(process)?
------------------------------------------------------------------------------
This case may happen when an application is developed by multiple teams as
submodules. If the all the submodules are linked into a single shared library,
then use the same version of MiAI Compute Engine will resolve this issue.
Ortherwise, different deployment models are contained in different shared
libraries, it's not required to use the same MiAI version but you should
controls the exported symbols from the shared library. This is actually a
best practice for all shared library, please read about GNU loader
version script for more details.
