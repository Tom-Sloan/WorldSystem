#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate data_simulator

# Make sure the script is executable
chmod +x storage/simulate_data.py

# Run the simulator
python storage/simulate_data.py