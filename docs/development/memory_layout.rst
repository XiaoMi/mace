Memory layout
==============

CPU runtime memory layout
--------------------------
The CPU tensor buffer is organized in the following order:

.. list-table::
    :header-rows: 1

    * - Tensor type
      - Buffer
    * - Intermediate input/output
      - NCHW
    * - Convolution Filter
      - OIHW
    * - Depthwise Convolution Filter
      - MIHW
    * - 1-D Argument, length = W
      - W

GPU runtime memory layout
--------------------------
GPU runtime implementation base on OpenCL, which uses 2D image with CL_RGBA
channel order as the tensor storage. This requires OpenCL 1.2 and above.

The way of mapping the Tensor data to OpenCL 2D image (RGBA) is critical for
kernel performance.

In CL_RGBA channel order, each 2D image pixel contains 4 data items.
The following tables describe the mapping from different type of tensors to
2D RGBA Image.

Input/Output Tensor
~~~~~~~~~~~~~~~~~~~~

The Input/Output Tensor is stored in NHWC format:

.. list-table::
    :header-rows: 1

    * - Tensor type
      - Buffer
      - Image size [width, height]
      - Explanation
    * - Channel-Major Input/Output
      - NHWC
      - [W * (C+3)/4, N * H]
      - Default Input/Output format
    * - Height-Major Input/Output
      - NHWC
      - [W * C, N * (H+3)/4]
      - WinogradTransform and MatMul output format
    * - Width-Major Input/Output
      - NHWC
      - [(W+3)/4 * C, N * H]
      - Unused now

Each Pixel of **Image** contains 4 elements. The below table list the
coordination relation between **Image** and **Buffer**.

.. list-table::
    :header-rows: 1

    * - Tensor type
      - Pixel coordinate relationship
      - Explanation
    * - Channel-Major Input/Output
      - P[i, j] = {E[n, h, w, c] | (n=j/H, h=j%H, w=i%W, c=[i/W * 4 + k])}
      - k=[0, 4)
    * - Height-Major Input/Output
      - P[i, j] = {E[n, h, w, c] | (n=j%N, h=[j/H*4 + k], w=i%W, c=i/W)}
      - k=[0, 4)
    * - Width-Major Input/Output
      - P[i, j] = {E[n, h, w, c] | (n=j/H, h=j%H, w=[i%W*4 + k], c=i/W)}
      - k=[0, 4)

Filter Tensor
~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Tensor
      - Buffer
      - Image size [width, height]
      - Explanation
    * - Convolution Filter
      - OIHW
      - [I, (O+3)/4 * W * H]
      - Convolution filter format，There is no difference compared to [H*W*I, (O+3)/4]
    * - Depthwise Convlution Filter
      - MIHW
      - [H * W * M, (I+3)/4]
      - Depthwise-Convolution filter format

Each Pixel of **Image** contains 4 elements. The below table list the
coordination relation between **Image** and **Buffer**.

.. list-table::
    :header-rows: 1

    * - Tensor type
      - Pixel coordinate relationship
      - Explanation
    * - Convolution Filter
      - P[m, n] = {E[o, i, h, w] | (o=[n/HW*4+k], i=m, h=T/W, w=T%W)}
      - HW= H * W, T=n%HW, k=[0, 4)
    * - Depthwise Convlution Filter
      - P[m, n] = {E[0, i, h, w] | (i=[n*4+k], h=m/W, w=m%W)}
      - only support multiplier == 1, k=[0, 4)

1-D Argument Tensor
~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Tensor type
      - Buffer
      - Image size [width, height]
      - Explanation
    * - 1-D Argument
      - W
      - [(W+3)/4, 1]
      - 1D argument format, e.g. Bias

Each Pixel of **Image** contains 4 elements. The below table list the
coordination relation between **Image** and **Buffer**.

.. list-table::
    :header-rows: 1

    * - Tensor type
      - Pixel coordinate relationship
      - Explanation
    * - 1-D Argument
      - P[i, 0] = {E[w] | w=i*4+k}
      - k=[0, 4)

