#!/bin/sh
pip install -r requirements.txt
python3 ./train-and-generate.py > training_status.html 2>&1 &
echo started training...
