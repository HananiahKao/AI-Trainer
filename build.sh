#!/bin/sh
git clone https://github.com/Homebrew/brew homebrew
eval "$(homebrew/bin/brew shellenv)"
brew update --force
chmod -R go-w "$(brew --prefix)/share/zsh"
brew install ollama
ollama pull mistral
pip3 install -r requirements.txt
