FROM cr.d.xiaomi.net/mace/mace-dev:latest

# Update source
# Looks like mirrors.163.com does not work in xiaomi network
# RUN sed -i 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/mirrors\.163\.com\/ubuntu\//g' /etc/apt/sources.list
RUN apt-get update -y

# Install gitlab runner
RUN curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-ci-multi-runner/script.deb.sh | bash
RUN apt-get install gitlab-ci-multi-runner

ENTRYPOINT gitlab-runner run
