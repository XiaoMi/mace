def if_hexagon_enabled(a):
    return select({
        "//micro:hexagon_enabled": a,
        "//conditions:default": [],
    })

def if_not_hexagon_enabled(a):
    return select({
        "//micro:hexagon_enabled": [],
        "//conditions:default": a,
    })

def new_local_repository_env_impl(repository_ctx):
    echo_cmd = "echo " + repository_ctx.attr.path
    echo_result = repository_ctx.execute(["bash", "-c", echo_cmd])
    src_path_str = echo_result.stdout.splitlines()[0]
    source_path = repository_ctx.path(src_path_str)

    work_path = repository_ctx.path(".")
    child_list = source_path.readdir()
    for child in child_list:
        child_name = child.basename
        repository_ctx.symlink(child, work_path.get_child(child_name))

    build_file_babel = Label("//:" + repository_ctx.attr.build_file)
    build_file_path = repository_ctx.path(build_file_babel)
    repository_ctx.symlink(build_file_path, work_path.get_child("BUILD"))

# a new_local_repository support environment variable
new_local_repository_env = repository_rule(
    implementation = new_local_repository_env_impl,
    local = True,
    attrs = {
        "path": attr.string(mandatory = True),
        "build_file": attr.string(mandatory = True),
    },
)
