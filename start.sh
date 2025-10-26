#!/bin/bash

echo "========================================"
echo "   RETAIL DATA WAREHOUSE SYSTEM"
echo "========================================"
echo

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo
echo "[2/3] Running production system..."
python run_production_system.py

echo
echo "[3/3] System completed!"
