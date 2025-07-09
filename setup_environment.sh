#!/bin/bash

echo "Setting up Python environment for Facial Classification Project..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found. Creating virtual environment..."
python3 -m venv facial_classification_env

echo "Activating virtual environment..."
source facial_classification_env/bin/activate

echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup completed successfully!"
echo
echo "To activate the environment in the future, run:"
echo "  source facial_classification_env/bin/activate"
echo
echo "To run the main program:"
echo "  python main.py" 