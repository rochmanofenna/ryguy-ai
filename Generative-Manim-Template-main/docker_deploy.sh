#!/bin/bash

# parse optional arg registry: '-r'
# Usage: sh docker_deploy.sh -r register.com:port
while getopts ":r:" opt
do
  case $opt in
    r)
      DOCKER_PUSH_REGISTRY=$OPTARG
      ;;
    ?)
      echo "invalid arg"
      exit 1
      ;;
  esac
done

# set image:tag
IMAGE_NAME=${DOCKER_IMAGE_NAME:-generative-manim/agent}
IMAGE_TAG=`git describe --tags`
DOCKER_PUSH_REGISTRY=${DOCKER_PUSH_REGISTRY:-localhost:5000}


# docker build
echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .
if [ $? -eq 0 ]; then
    echo "succeed"
else
    echo "error failed ！！！"
    exit 1
fi

echo "Build docker image ${IMAGE_NAME}:${IMAGE_TAG} done."

# docker tag with repo
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_PUSH_REGISTRY}/${IMAGE_NAME}:latest
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_PUSH_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# docker push
echo "Push docker images to ${DOCKER_PUSH_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push ${DOCKER_PUSH_REGISTRY}/${IMAGE_NAME}:latest
docker push ${DOCKER_PUSH_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
if [ $? -eq 0 ]; then
    echo "succeed"
else
    echo "error failed ！！！"
    exit 1
fi
echo "Push docker images done."