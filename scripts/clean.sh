#!/bin/bash

echo "Cleaning generated files..."

if [ -d "outputs" ]; then
    rm -rf outputs/*
    echo "Cleared outputs/"
fi

find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.log" -delete

echo "Removing dangling Docker images..."
docker image prune -f

echo "Cleanup complete."