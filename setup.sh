#!/bin/bash

# Define the name of the virtual environment
VENV_NAME="jtk"

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install it first."
    exit 1
fi

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "Error: virtualenv is not installed. Please install it first."
    exit 1
fi

# Check if the virtual environment already exists
if [ -d "$VENV_NAME" ]; then
    echo "Error: Virtual environment '$VENV_NAME' already exists."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment................"
python3 -m venv  $VENV_NAME

# Activate the virtual environment
echo "Activating virtual environment................"
source $VENV_NAME/bin/activate

# Install dependencies from requirements.txt
echo "Installing requirements................"
pip install --upgrade pip
pip install -r requirements.txt
echo "Virtual environment '$VENV_NAME' setup is complete."