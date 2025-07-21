#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name of the virtual environment directory
VENV_DIR="env"

# --- 1. Create a Python virtual environment ---
echo ">>> Creating virtual environment in '$VENV_DIR'..."
python3 -m venv $VENV_DIR
echo ">>> Virtual environment created."

# --- 2. Activate the virtual environment and install requirements ---
echo ">>> Activating virtual environment and installing requirements..."
# Note: 'source' is a shell command, so we call pip directly from the venv path
$VENV_DIR/bin/pip install -r requirements.txt

# Install the local vit-pytorch package in editable mode
echo ">>> Installing local vit-pytorch package..."
$VENV_DIR/bin/pip install -e vit-pytorch/
echo ">>> Requirements installed successfully."

# --- 3. Run the Python script ---
echo ">>> Running the Tokenformer training script..."
$VENV_DIR/bin/python tokenformer.py
echo ">>> Script finished."

# --- 4. Deactivation Information ---
echo -e "\n---"
echo "To manually activate this environment in your terminal, run:"
echo "source $VENV_DIR/bin/activate"
echo "---"
