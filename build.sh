#!/bin/sh
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
pip3 install -r requirements.txt
