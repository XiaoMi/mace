OpenCL Image Storage Layout
===========================

Use **Image** object to optimize memory access and parallel computing based on OpenCL 2.0.


Design the corresponding **Image** format to optimize memory access for different Op algorithm.
Each pixel of **Image** object contains 4 elements(e.g. RGBA).


The Followings are the **Buffer** and **Image** format for all **Tensors**.

Input/Output
---
**Mace** use NHWC format Input/Output.

| Tensor| Buffer| Image Size [Width, Height]| Explanation|
| --------- | :---------:|:--------:|:----:|
|Channel-Major Input/Output | NHWC | [W * (C+3)/4, N * H] | Default Input/Output format|
|Height-Major Input/Output | NHWC | [W * C, N * (H+3)/4] | Winograd Convolution format| 
|Width-Major Input/Output | NHWC | [(W+3)/4 * C, N * H] | Winograd Convolution format|

Each Pixel of **Image** contains 4 elements. The below table list the coordination relation 
between **Image** and **Buffer**.

| Tensor| Pixel Coordinate Relation| Explanation
| --------- | :---------:| :-----: |
|Channel-Major Input/Output | P[i, j] = {E[n, h, w, c] &#124; (n=j/H, h=j%H, w=i%W, c=[i/W * 4 + k])}| k=[0, 4)|
|Height-Major Input/Output | P[i, j] = {E[n, h, w, c] &#124; (n=j%N, h=[j/H*4 + k], w=i%W, c=i/W)}| k=[0, 4)|
|Width-Major Input/Output | P[i, j] = {E[n, h, w, c] &#124; (n=j/H, h=j%H, w=[i%W*4 + k], c=i/W)}| k=[0, 4)|


Filter
---
| Tensor| Buffer| Image Size [Width, Height]| Explanation|
| --------- | :---------:|:--------:|:----:|
|Convolution Filter | HWOI | [RoundUp<4>(I), H * W * (O+3)/4]|Convolution filter format，There is no difference compared to [H*w*I, (O+3)/4]|
|Depthwise Convlution Filter | HWIM | [H * W * M, (I+3)/4]|Depthwise-Convolution filter format|

Each Pixel of **Image** contains 4 elements. The below table list the coordination relation 
between **Image** and **Buffer**.

| Tensor| Pixel Coordinate Relation| Explanation|
| --------- | :---------:| :-----:|
|Convolution Filter | P[m, n] = {E[h, w, o, i] &#124; (h=T/W, w=T%W, o=[n/HW*4+k], i=m)}| HW= H * W, T=n%HW, k=[0, 4)|
|Depthwise Convlution Filter | P[m, n] = {E[h, w, i, 0] &#124; (h=m/W, w=m%W, i=[n*4+k])}| only support multiplier == 1, k=[0, 4)| 

1-D Argument
---
| Tensor| Buffer| Image Size [Width, Height]| Explanation|
| --------- | :---------:|:--------:|:----:|
|1-D Argument | W | [(W+3)/4, 1] | 1D argument format, e.g. Bias|

Each Pixel of **Image** contains 4 elements. The below table list the coordination relation 
between **Image** and **Buffer**.

| Tensor| Pixel Coordinate Relation| Explanation|
| --------- | :---------:| :-----:|
|1-D Argument | P[i, 0] = {E[w] &#124; w=i*4+k}| k=[0, 4)|
