FROM registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev:latest

# Update source
RUN apt-get update -y

# Install gitlab runner
RUN curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-ci-multi-runner/script.deb.sh | bash
RUN apt-get install gitlab-ci-multi-runner

# set timezone
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

ENTRYPOINT gitlab-runner run
