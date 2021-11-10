#!/bin/bash

IMAGE_NAME=xiaolao/paddle-launcher:latest

echo "Building image $IMAGE_NAME"
docker build -t ${IMAGE_NAME} .

echo "Pushing image $IMAGE_NAME"
docker push ${IMAGE_NAME}
