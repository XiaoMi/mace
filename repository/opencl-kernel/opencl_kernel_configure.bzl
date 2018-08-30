"""Repository rule for opencl encrypt kernel autoconfiguration, borrow from tensorflow
"""
def _opencl_encrypt_kernel_impl(repository_ctx):
  repository_ctx.template(
      "BUILD",
      Label("//repository/opencl-kernel:BUILD.tpl"))

  mace_root_path = str(repository_ctx.path(Label("@mace//:BUILD")))[:-len("BUILD")]
  generated_files_path = repository_ctx.path("gen")

  ret = repository_ctx.execute(
      ["test", "-f", "%s/.git/logs/HEAD" % mace_root_path])
  if ret.return_code == 0:
    unused_var = repository_ctx.path(Label("//:.git/HEAD"))
  ret = repository_ctx.execute(
      ["test", "-f", "%s/.git/refs/heads/master" % mace_root_path])
  if ret.return_code == 0:
    unused_var = repository_ctx.path(Label("//:.git/refs/heads/master"))

  ret = repository_ctx.execute(
      ["test", "-f", "%s/mace/kernels/opencl/cl/common.h" % mace_root_path])
  if ret.return_code == 0:
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/activation.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/addn.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/batch_norm.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/bias_add.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/buffer_to_image.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/channel_shuffle.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/common.h"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/concat.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/conv_2d.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/conv_2d_1x1.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/conv_2d_3x3.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/crop.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/deconv_2d.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/depth_to_space.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/depthwise_conv2d.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/eltwise.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/fully_connected.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/matmul.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/pad.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/pooling.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/reduce_mean.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/resize_bicubic.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/resize_bilinear.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/split.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/softmax.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/space_to_batch.cl"))
    unused_var = repository_ctx.path(Label("//:mace/kernels/opencl/cl/winograd_transform.cl"))

  python_bin_path = repository_ctx.which("python")

  repository_ctx.execute([
      python_bin_path, '%s/mace/python/tools/encrypt_opencl_codegen.py' % mace_root_path,
      '--cl_kernel_dir=%s/mace/kernels/opencl/cl' % mace_root_path,
      '--output_path=%s/encrypt_opencl_kernel' % generated_files_path
  ], quiet=False)


encrypt_opencl_kernel_repository = repository_rule(
    implementation = _opencl_encrypt_kernel_impl,
)
