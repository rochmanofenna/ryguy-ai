#!/bin/bash

IMAGE_TAG=latest
if [ -n "$1" ]; then
IMAGE_TAG="$1"
fi
echo "IMAGE_TAG:$IMAGE_TAG"

DOCKER_PULL_REGISTRY=${DOCKER_PULL_REGISTRY:-localhost:5000}
IMAGE_NAME=${DOCKER_IMAGE_NAME:-generative-manim/agent}
SERVICE_NAME=${SERVICE_NAME:-"generative-manim-service"}
HOST_PORT=${HOST_PORT:-8089}
DOCKER_PORT=${DOCKER_PORT:-8089}
DOCKER_VOLUME_SRC=${DOCKER_VOLUME_SRC:-$(pwd)}
DOCKER_VOLUME_DST=/config

# pull image
echo "docker pull ${DOCKER_PULL_REGISTRY}/${IMAGE_NAME}:$IMAGE_TAG"
docker pull ${DOCKER_PULL_REGISTRY}/${IMAGE_NAME}:$IMAGE_TAG
if [ $? -eq 0 ]; then
    echo "succeed"
else
    echo "error failed ！！！"
    exit 1
fi

echo "docker image pull done"

# stop old container
if docker ps -a --format '{{.Names}}' | grep -Eq "^${SERVICE_NAME}\$"; then
    docker stop $SERVICE_NAME
    docker rm $SERVICE_NAME
fi

# docker run
echo "docker run..."
docker run --add-host=host.docker.internal:host-gateway -d --restart unless-stopped --env-file ${ENV_FILE:-.env} --name $SERVICE_NAME \
	-p $HOST_PORT:$DOCKER_PORT \
        ${DOCKER_PULL_REGISTRY}/$IMAGE_NAME:$IMAGE_TAG
if [ $? -eq 0 ]; then
    echo "succeed"
else
    echo "error failed ！！！"
    exit 1
fi
echo "docker run done"