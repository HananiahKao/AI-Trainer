#!/bin/sh
pip install -r requirements.txt
git config --global user.name "HananiahKao"
git config --global user.email "HananiahKao@users.noreply.github.com"
./fetch_github_repos.sh &
python3 ./train-and-generate.py &
echo started training...
