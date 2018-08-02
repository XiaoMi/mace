# v0.9.0 (2018-07-20)
------
## Improvements
1. New work flow and documents.
2. Separate the model library from MACE library.
3. Reduce the size of static and dynamic library.
4. Support `ArgMax` Operations.
5. Support `Deconvolution` of Caffe.
6. Support NDK-17b.

## Incompatible Changes
1. Use file to store OpenCL tuned parameters and Add `SetOpenCLParameterPath` API.

## New APIs
1. Add a new `MaceEngine::Init` API with model data file.

## Bug Fixed
1. Not unmap the model data file when load model from files with CPU runtime.
2. 2D LWS tuning does not work.
3. Winograd convolution of GPU failed when open tuning.
4. Incorrect dynamic library of host.

## Acknowledgement
Appreciate for the following guys contribute code to make MACE better.

Zero King(@l2dy), James Bie(@JamesBie), Sun Aries(@SunAriesCN), Allen(@allen0125),
conansherry(@conansherry), 黎明灰烬(@jackwish)


# v0.8.0 (2018-05-31)
------
1. Change build and run tools
2. Handle runtime failure

# v0.7.0 (2018-05-18)
------
1. Change interface that report error type
2. Improve CPU performance
3. Merge CPU/GPU engine to on

# v0.6.3 (2018-05-21)
------
1. support `float` `data_type` when running in GPU


# v0.6.2 (2018-05-17)
------
* Return status instead of abort when allocate failed


# v0.6.0 (2018-04-04)
------
1. Change mace header interfaces, only including necessary methods.

