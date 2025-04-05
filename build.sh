#!/bin/sh
pip install -r requirements.txt
git config --global user.name "HananiahKao"
git config --global user.email "HananiahKao@users.noreply.github.com"
./fetch_github_repos.sh 2>&1 > fetch_repo_status.html &
python3 ./train-and-generate.py 2>&1 > training_status.html &
echo started training...
