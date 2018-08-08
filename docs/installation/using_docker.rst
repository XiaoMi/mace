Using docker
=============

Pull or build docker image
---------------------------

MACE provides docker images with dependencies installed and also Dockerfiles for images building,
you can pull the existing ones directly or build them from the Dockerfiles.
In most cases, the ``lite edition`` image can satisfy developer's basic needs.

.. note::
    It's highly recommended to pull built images.

- ``lite edition`` docker image.

.. code:: sh

    # Pull lite edition docker image
    docker pull registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite
    # Build lite edition docker image
    docker build -t registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite ./docker/mace-dev-lite

- ``full edition`` docker image (which contains multiple NDK versions and other dev tools).

.. code:: sh

    # Pull full edition docker image
    docker pull registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev
    # Build full edition docker image
    docker build -t registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev ./docker/mace-dev

.. note::

    We will show steps with lite edition later.


Using the image
-----------------

Create container with the following command

.. code:: sh

    # Create a container named `mace-dev`
    docker run -it --privileged -d --name mace-dev \
               -v /dev/bus/usb:/dev/bus/usb --net=host \
               -v /local/path:/container/path \
               -v /usr/bin/docker:/usr/bin/docker \
               -v /var/run/docker.sock:/var/run/docker.sock \
               registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite
    # Execute an interactive bash shell on the container
    docker exec -it mace-dev /bin/bash
