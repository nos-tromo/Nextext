#!/usr/bin/env sh

set -eu

for volume_name in \
    huggingface-cache \
    nltk-cache \
    ollama-cache \
    spacy-cache \
    torch-cache
do
    docker volume create "${volume_name}" >/dev/null
    printf "Ensured Docker volume exists: %s\n" "${volume_name}"
done
