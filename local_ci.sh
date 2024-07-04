#!/bin/bash
set -e

echo "Installing dependencies..."
#pip install -r requirements.txt

echo "Running black..."
black .

echo "Running flake8..."
flake8 . --ignore=E501

echo "Running mypy..."
mypy . --ignore-missing-imports

echo "Running tests..."
python -m unittest discover tests