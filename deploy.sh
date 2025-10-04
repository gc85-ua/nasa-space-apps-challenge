#!/bin/bash

ROOT_DIR="nasa-space-apps-challenge"

current_dir="$(pwd | rev | cut -d'/' -f1 | rev)"

if [ "$current_dir" != "$ROOT_DIR" ]; then
  echo "Please run this script from the root directory of the project: $ROOT_DIR"
  exit 1
fi

if [ ! "$(which docker)" ]; then
  echo "Docker is not installed. Please install Docker and try again."
  exit 1
fi

if [ ! -f "frontend.dockerfile" ]; then
    echo "frontend.dockerfile not found!"
    exit 1
fi

if [ ! -f "api.dockerfile" ]; then
    echo "api.dockerfile not found!"
    exit 1
fi

if [ ! -f "docker-compose.yml" ]; then
    echo "docker-compose.yml not found!"
    exit 1
fi

echo "Building frontend Docker image..."

if ! docker build -t frontend:latest -f frontend.dockerfile . ; then
    echo "Failed to build frontend Docker image."
    exit 1
fi

echo "Building API Docker image..."

if ! docker build -t api:latest -f api.dockerfile . ; then
    echo "Failed to build API Docker image."
    exit 1
fi

echo "Frontend and API Docker images built successfully."
echo "Starting containers with docker-compose..."

if ! docker-compose up -d ; then
    echo "Failed to start containers with docker-compose."
    exit 1
fi

echo "Containers started successfully."