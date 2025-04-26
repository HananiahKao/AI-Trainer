#!/bin/sh
git clone https://github.com/Homebrew/brew homebrew
eval "$(homebrew/bin/brew shellenv)"
brew update --force
chmod -R go-w "$(brew --prefix)/share/zsh"
brew install ollama
ollama serve &
ollama pull mistral
pip install -r requirements.txt
