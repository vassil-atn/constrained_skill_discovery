ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# # Install whatever you need as additional dependencies.
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/exts/constrained_skill_discovery && \
    pip install -e ."

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace

# install our extensions
RUN cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/exts/skrl && \
${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install -e .


RUN  echo "git config --global --add safe.directory ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}" >> ${HOME}/.bashrc
