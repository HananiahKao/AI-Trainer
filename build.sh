#!/bin/sh
pip install -r requirements.txt
git config --global user.name "HananiahKao"
git config --global user.email "HananiahKao@users.noreply.github.com"
./fetch_github_repos.sh > fetch_repo_status.html 2>&1 &
python3 ./train-and-generate.py > training_status.html 2>&1 &
echo started training...
