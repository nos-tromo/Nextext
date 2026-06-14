#!/usr/bin/env sh

set -eu

# nltk-cache / spacy-cache back the language-resource preloads; ollama-cache
# serves operators running a sibling Ollama container on inference-net (see
# README). Model inference itself happens on external endpoints — no model
# weight caches are needed anymore.
for volume_name in \
    nltk-cache \
    ollama-cache \
    spacy-cache
do
    docker volume create "${volume_name}" >/dev/null
    printf "Ensured Docker volume exists: %s\n" "${volume_name}"
done
