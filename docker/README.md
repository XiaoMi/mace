Development Docker Image
========

* Login in Xiaomi Docker Hub

  ```
  docker login docker.d.xiaomi.net
  ```

* Build the image

  ```
  docker build -t docker.d.xiaomi.net/mace/mace-dev .
  ```

* Push the image to Xiaomi Docker Hub

  ```
  docker push docker.d.xiaomi.net/mace/mace-dev
  ```

* Pull and run the image

  ```
  # Set 'host' network to use ADB
  docker run -it --net=host docker.d.xiaomi.net/mace/mace-dev /bin/bash
  ```
