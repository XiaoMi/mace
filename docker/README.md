# 包含mace环境的docker镜像
========

* Login in [小米容器仓库](http://docs.api.xiaomi.net/docker-registry/)

  ```
  docker login cr.d.xiaomi.net
  ```

* 使用`Dockerfile`编译镜像

  ```
  docker build -t cr.d.xiaomi.net/mace/mace-dev .
  ```

* 或者从镜像仓库直接pull镜像

  ```
  docker push cr.d.xiaomi.net/mace/mace-dev
  ```

* 启动容器

  ```
  # Set 'host' network to use ADB
  docker run -it --rm -v /local/path:/container/path --net=host cr.d.xiaomi.net/mace/mace-dev /bin/bash
  ```
