#!/bin/sh
set -e

ollama serve &

# Wait until Ollama server is responsive
until curl -s -o /dev/null http://localhost:11434; do
  echo "Waiting for Ollama server to be ready..."
  sleep 1
done

ollama create app -f /root/.ollama/models/Modelfile

tail -f /dev/null
