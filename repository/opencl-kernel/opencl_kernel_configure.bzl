"""Repository rule for opencl encrypt kernel autoconfiguration, borrow from tensorflow
"""
def _opencl_encrypt_kernel_impl(repository_ctx):
  repository_ctx.template(
      "BUILD",
      Label("//repository/opencl-kernel:BUILD.tpl"))

  mace_root_path = str(repository_ctx.path(Label("@mace//:BUILD")))[:-len("BUILD")]

  generated_files_path = repository_ctx.path("gen")

  python_bin_path = repository_ctx.which("python")

  repository_ctx.execute([
      python_bin_path, '%s/mace/python/tools/encrypt_opencl_codegen.py' % mace_root_path,
      '--cl_kernel_dir=%s/mace/kernels/opencl/cl' % mace_root_path,
      '--output_path=%s/encrypt_opencl_kernel' % generated_files_path
  ], quiet=False)


encrypt_opencl_kernel_repository = repository_rule(
    implementation = _opencl_encrypt_kernel_impl,
    local=True,
)
