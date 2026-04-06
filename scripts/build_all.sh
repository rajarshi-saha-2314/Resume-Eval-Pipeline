#!/bin/bash

set -e

echo "Building Docker images..."

docker build -t logistic_regression ./models/logistic_regression
docker build -t random_forest ./models/random_forest

echo "Docker images built successfully."