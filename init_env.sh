#!/bin/bash

if command -v uv &> /dev/null; then
    echo "Using uv for fast installation..."
    uv sync
    uv run python -m spacy download en_core_web_lg
else
    echo "uv not found, falling back to pip..."
    pip install -r dependencies.txt
    python -m spacy download en_core_web_lg
fi