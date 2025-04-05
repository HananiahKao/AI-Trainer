#!/bin/zsh
pip install -r requirements.txt
./fetch_github_repos.sh &
python3 ./train-and-generate.py &
echo started training...
