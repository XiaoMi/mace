Docker Images
=============

-  Login in `Xiaomi Docker
   Registry <http://docs.api.xiaomi.net/docker-registry/>`__

``docker login cr.d.xiaomi.net``

-  Build with ``Dockerfile``

``docker build -t cr.d.xiaomi.net/mace/mace-dev .``

-  Pull image from docker registry

``docker pull cr.d.xiaomi.net/mace/mace-dev``

-  Create container

``# Set 'host' network to use ADB   docker run -it --rm -v /local/path:/container/path --net=host cr.d.xiaomi.net/mace/mace-dev /bin/bash``
