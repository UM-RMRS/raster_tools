#!/bin/env bash
mkdir -p ../release/
rm -rf ../release/raster_tools
git clone . ../release/raster_tools
cd ../release/raster_tools
rm -rf .flake8
rm -rf .git/
rm -rf test/
rm .gitignore
rm .pre-commit-config.yaml
cd ..
tar -czvf raster_tools.tar.gz raster_tools
