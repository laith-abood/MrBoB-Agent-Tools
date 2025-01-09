#!/bin/bash

# Exit on error
set -e

echo "Setting up development environment..."

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt

echo "Development environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
