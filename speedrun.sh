#!/bin/bash

echo "---------------------------------------"
echo "NanoJiT Speedrun: Setup"
echo "---------------------------------------"
pip install -e .

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "---------------------------------------"
echo "NanoJiT Speedrun: GPU Check"
echo "---------------------------------------"
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

echo "---------------------------------------"
echo "NanoJiT Speedrun: Training"
echo "---------------------------------------"
python src/nanojit/train.py